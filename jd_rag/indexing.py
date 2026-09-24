from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings
from .events import ProgressCallback, ignore_progress

if TYPE_CHECKING:
    from langchain_chroma import Chroma


@dataclass
class UpdateSummary:
    new: int = 0
    modified: int = 0
    deleted: int = 0
    indexed: dict[str, int] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class Indexer:
    """Incremental indexing. Do not run overlapping updates on the same database."""
    def __init__(self, settings: Settings, report: ProgressCallback = ignore_progress,
                 processor=None, splitter=None):
        self.settings = settings
        self.report = report
        self._processor = processor
        self._splitter = splitter

    @property
    def processor(self):
        if self._processor is None:
            from .documents import DocumentProcessor
            self._processor = DocumentProcessor(self.settings, self.report)
        return self._processor

    @property
    def splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return self._splitter

    def prepare_storage(self) -> bool:
        """Validate storage and return whether a database already exists."""
        self.settings.notes_dir.mkdir(parents=True, exist_ok=True)
        self.settings.db_dir.parent.mkdir(parents=True, exist_ok=True)
        self.settings.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        db_exists = self.settings.db_dir.exists() and any(self.settings.db_dir.iterdir())
        if db_exists and not self.settings.manifest_file.exists():
            raise RuntimeError(
                "Existing chroma_db has no indexed_files.json. "
                "For a clean start, back up/delete chroma_db and indexed_files.json."
            )
        if not db_exists and self.settings.manifest_file.exists():
            self.report("Database missing: removing stale indexing history.")
            self.settings.manifest_file.unlink()
        if db_exists:
            self.load_manifest()
        return db_exists

    def open_database(self):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        return Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=OllamaEmbeddings(model=self.settings.embed_model),
            persist_directory=str(self.settings.db_dir),
        )

    def file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_manifest(self) -> dict:
        if not self.settings.manifest_file.exists():
            return {"pipeline_version": self.settings.pipeline_version, "files": {}}
        with self.settings.manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (not isinstance(manifest, dict)
                or manifest.get("pipeline_version") != self.settings.pipeline_version
                or not isinstance(manifest.get("files"), dict)):
            raise RuntimeError(
                "The existing manifest belongs to an older indexing pipeline. "
                "Back up and delete BOTH chroma_db and indexed_files.json, then restart."
            )
        return manifest

    def save_manifest(self, manifest: dict) -> None:
        temp = self.settings.manifest_file.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(temp, self.settings.manifest_file)

    def scan_files(self) -> dict[str, str]:
        current = {}
        for path in sorted(self.settings.notes_dir.rglob("*")):
            if (path.is_file() and path.suffix.lower() in self.settings.supported
                    and not path.name.startswith("~$")
                    and not path.name.startswith(".")):
                current[str(path.resolve())] = self.file_hash(path)
        return current

    def file_chunk_ids(self, db: Chroma, source: str) -> list[str]:
        return db.get(where={"source": source}, include=[])["ids"]

    def index_file(self, db: Chroma, path: Path, expected_hash: str) -> int:
        """Stage replacement chunks before removing old chunks; roll back on errors."""
        source = str(path.resolve())
        docs = self.processor.load_single_document(path)
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No extractable content found (or no readable images).")
        if self.file_hash(path) != expected_hash:
            raise RuntimeError("File changed during processing; retry the update.")
        new_ids = [hashlib.sha256(
            f"{self.settings.pipeline_version}|{source}|{expected_hash}|{i}".encode("utf-8")
        ).hexdigest() for i in range(len(chunks))]
        old_ids = self.file_chunk_ids(db, source)
        # Clear any leftover partial attempt with the same stable IDs.
        existing_new = set(db.get(ids=new_ids, include=[])["ids"])
        if existing_new:
            db.delete(ids=list(existing_new))
        added = []
        try:
            self.report(f"  Embedding {len(chunks)} chunks (batches of {self.settings.batch_size})...")
            for i in range(0, len(chunks), self.settings.batch_size):
                batch_ids = new_ids[i:i + self.settings.batch_size]
                db.add_documents(documents=chunks[i:i + self.settings.batch_size], ids=batch_ids)
                added.extend(batch_ids)
                self.report(f"  Indexed {min(i + self.settings.batch_size, len(chunks))}/{len(chunks)}")
            if self.file_hash(path) != expected_hash:
                raise RuntimeError("File changed during indexing; retaining previous version.")
            stale_ids = [ident for ident in old_ids if ident not in new_ids]
            if stale_ids:
                db.delete(ids=stale_ids)
            return len(chunks)
        except Exception:
            # Existing previous-version chunks were not deleted yet.
            if added:
                db.delete(ids=added)
            raise

    def update_database(self, db: Chroma) -> UpdateSummary:
        manifest = self.load_manifest()
        previous = manifest["files"]
        current = self.scan_files()
        new = [s for s in current if s not in previous]
        changed = [s for s in current if s in previous and current[s] != previous[s]]
        deleted = [s for s in previous if s not in current]
        summary = UpdateSummary(new=len(new), modified=len(changed), deleted=len(deleted))
        self.report(f"\nChanges: {len(new)} new, {len(changed)} modified, {len(deleted)} deleted")
        if not (new or changed or deleted):
            self.report("Database already up to date.")
            return summary
        for source in new + changed:
            path = Path(source)
            self.report(f"\nProcessing: {path.name}")
            try:
                count = self.index_file(db, path, current[source])
                previous[source] = current[source]
                self.save_manifest(manifest)
                summary.indexed[source] = count
                self.report(f"  Complete: {count} chunks")
            except Exception as exc:
                summary.errors[source] = str(exc)
                self.report(f"  ERROR: {exc}")
                self.report("  This file will be retried on the next update.")
        for source in deleted:
            self.report(f"\nRemoving deleted file: {Path(source).name}")
            try:
                ids = self.file_chunk_ids(db, source)
                if ids:
                    db.delete(ids=ids)
                previous.pop(source, None)
                self.save_manifest(manifest)
                summary.removed.append(source)
                self.report(f"  Removed {len(ids)} chunks")
            except Exception as exc:
                summary.errors[source] = str(exc)
                self.report(f"  ERROR: {exc}; will retry on next update.")
        self.report("\nDatabase update complete.")
        return summary

    def update_and_unload(self, db: Chroma) -> UpdateSummary:
        try:
            return self.update_database(db)
        finally:
            # Release Docling objects and unload Ollama's vision model.
            if self._processor is not None:
                self._processor.clear_cache()
                gc.collect()
                self._processor.unload_vision()
