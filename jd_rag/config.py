from dataclasses import dataclass
from pathlib import Path

# The project root, NOT the jd_rag package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Settings:
    base_dir: Path = PROJECT_ROOT
    notes_dir: Path | None = None
    db_dir: Path | None = None
    manifest_file: Path | None = None
    llm_model: str = "qwen3:4b-instruct"
    embed_model: str = "nomic-embed-text"
    vision_model: str = "granite3.2-vision"
    collection_name: str = "my_notes"
    batch_size: int = 4
    retrieval_k: int = 4
    pipeline_version: str = "docling-granite-v1"
    supported: frozenset[str] = frozenset({".txt", ".md", ".pdf", ".pptx", ".docx"})
    render_full_slides: bool = True
    max_visuals_per_file: int = 80

    def __post_init__(self):
        base = Path(self.base_dir).expanduser().resolve()
        object.__setattr__(self, "base_dir", base)
        for name, default in (("notes_dir", "my_notes"), ("db_dir", "chroma_db"),
                              ("manifest_file", "indexed_files.json")):
            value = getattr(self, name)
            path = Path(value).expanduser() if value is not None else base / default
            if not path.is_absolute():
                path = base / path
            object.__setattr__(self, name, path.resolve())
        if self.batch_size < 1 or self.retrieval_k < 1:
            raise ValueError("Batch size and retrieval depth must be positive.")
