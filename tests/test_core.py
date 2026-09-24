"""Core contract tests; no Ollama server or third-party packages required."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from jd_rag.config import Settings, PROJECT_ROOT
from jd_rag.indexing import Indexer
from jd_rag.models import list_chat_models
from jd_rag.chat import ChatSession, format_context
import JD_RAG


class FakeDB:
    def __init__(self):
        self.rows = {}
        self.calls = []
        self.fail_on = None

    def get(self, *, where=None, ids=None, include=None):
        return {'ids': [key for key, doc in self.rows.items()
                        if (ids is None or key in ids)
                        and (where is None or doc.metadata['source'] == where['source'])]}

    def delete(self, *, ids):
        for key in ids:
            self.rows.pop(key, None)

    def add_documents(self, *, documents, ids):
        self.calls.append(len(ids))
        if len(self.calls) == self.fail_on:
            raise RuntimeError('Embedding service unavailable')
        self.rows.update(zip(ids, documents))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(base_dir=Path(self.temp.name))
        self.processor = Mock()
        self.processor.load_single_document.side_effect = lambda p: [
            NS(page_content=p.read_text(), metadata={'source': str(p.resolve())})]
        self.splitter = Mock()
        self.splitter.split_documents.side_effect = lambda docs: docs
        self.messages = []
        self.indexer = Indexer(self.settings, self.messages.append, self.processor, self.splitter)
        self.indexer.prepare_storage()
        self.db = FakeDB()

    def note(self, text='original'):
        path = self.settings.notes_dir / 'notes.txt'
        path.write_text(text)
        return path

    def test_project_paths_and_overrides(self):
        self.assertEqual(Settings().notes_dir, PROJECT_ROOT / 'my_notes')
        self.assertEqual(Settings().pipeline_version, 'docling-granite-v1')
        changed = Settings(base_dir=self.settings.base_dir, notes_dir=Path('library'))
        self.assertEqual(changed.notes_dir, self.settings.base_dir / 'library')

    def test_new_unchanged_modified_deleted(self):
        path = self.note()
        first = self.indexer.update_and_unload(self.db)
        self.assertEqual(first.new, 1)
        source = str(path.resolve())
        expected = hashlib.sha256(f'docling-granite-v1|{source}|{self.indexer.file_hash(path)}|0'.encode()).hexdigest()
        self.assertEqual(list(self.db.rows), [expected])
        second = self.indexer.update_and_unload(self.db)
        self.assertEqual((second.new, second.modified, second.deleted), (0, 0, 0))
        self.processor.load_single_document.assert_called_once()
        path.write_text('changed')
        self.assertEqual(self.indexer.update_and_unload(self.db).modified, 1)
        self.assertNotIn(expected, self.db.rows)
        self.assertEqual(len(self.db.rows), 1)
        path.unlink()
        summary = self.indexer.update_and_unload(self.db)
        self.assertEqual(summary.removed, [source])
        self.assertFalse(self.db.rows)
        self.assertEqual(self.indexer.load_manifest()['files'], {})

    def test_failed_extraction_preserves_previous_and_retries(self):
        path = self.note()
        self.indexer.update_and_unload(self.db)
        before = dict(self.db.rows)
        manifest = self.indexer.load_manifest()
        path.write_text('modified')
        self.processor.load_single_document.side_effect = RuntimeError('conversion failed')
        result = self.indexer.update_and_unload(self.db)
        self.assertIn(str(path.resolve()), result.errors)
        self.assertEqual(self.db.rows, before)
        self.assertEqual(self.indexer.load_manifest(), manifest)
        self.assertEqual(self.indexer.update_database(self.db).modified, 1)
        self.processor.clear_cache.assert_called()
        self.processor.unload_vision.assert_called()

    def test_batch_failure_rolls_back_added_chunks(self):
        path = self.note()
        self.indexer.update_database(self.db)
        before = dict(self.db.rows)
        path.write_text('new text')
        self.splitter.split_documents.side_effect = lambda docs: docs * 9
        self.db.calls.clear()
        self.db.fail_on = 2
        result = self.indexer.update_database(self.db)
        self.assertTrue(result.errors)
        self.assertEqual(self.db.calls, [4, 4])
        self.assertEqual(self.db.rows, before)

    def test_batch_sizes(self):
        self.note()
        self.splitter.split_documents.side_effect = lambda docs: docs * 9
        self.indexer.update_database(self.db)
        self.assertEqual(self.db.calls, [4, 4, 1])

    def test_file_changed_during_processing(self):
        path = self.note()
        expected = self.indexer.file_hash(path)
        path.write_text('changed mid-update')
        with self.assertRaisesRegex(RuntimeError, 'File changed'):
            self.indexer.index_file(self.db, path, expected)
        self.assertFalse(self.db.rows)

    def test_manifest_version_validation(self):
        self.indexer.save_manifest({'pipeline_version':'other', 'files':{}})
        with self.assertRaises(RuntimeError):
            self.indexer.load_manifest()

    def test_existing_database_without_manifest_rejected(self):
        self.settings.db_dir.mkdir()
        (self.settings.db_dir / 'placeholder').touch()
        with self.assertRaisesRegex(RuntimeError, 'no indexed_files'):
            self.indexer.prepare_storage()

    def test_stale_manifest_removed(self):
        self.indexer.save_manifest({'pipeline_version':'docling-granite-v1','files':{}})
        self.assertFalse(self.indexer.prepare_storage())
        self.assertFalse(self.settings.manifest_file.exists())

    def test_scan_recurses_and_filters_files(self):
        self.note()
        nested = self.settings.notes_dir / 'nested'
        nested.mkdir()
        for name in ['paper.PDF', '~$draft.docx', '.hidden.txt', 'ignore.png']:
            (nested / name).write_text('data')
        self.assertEqual({Path(s).name for s in self.indexer.scan_files()}, {'notes.txt','paper.PDF'})

    def test_models_filter_and_default(self):
        client = Mock()
        names = ['llama:latest','qwen3:4b-instruct','nomic-embed-text:latest','granite3.2-vision:latest','embed:latest']
        client.list.return_value = NS(models=[NS(model=name) for name in names])
        client.show.side_effect = lambda n: {'capabilities':['embedding' if n=='embed:latest' else 'completion']}
        models = list_chat_models(self.settings, client)
        self.assertEqual(models.names, ['llama:latest','qwen3:4b-instruct'])
        self.assertEqual(models.default, 'qwen3:4b-instruct')

    def test_models_offline_empty_and_missing_default(self):
        client = Mock()
        client.list.side_effect = RuntimeError('offline')
        result = list_chat_models(self.settings, client)
        self.assertEqual(result.default, self.settings.llm_model)
        self.assertEqual(result.error, 'offline')
        client.list.side_effect = None
        client.list.return_value = {'models':[]}
        self.assertEqual(list_chat_models(self.settings, client).default, self.settings.llm_model)
        client.list.return_value = {'models':[{'name':'llama:latest'}]}
        client.show.return_value = {}
        self.assertEqual(list_chat_models(self.settings, client).default, 'llama:latest')

    def test_terminal_choices(self):
        available = NS(names=['llama:latest','qwen3:4b-instruct'], default='qwen3:4b-instruct', error=None)
        for values, expected in [([''],available.default), (['1'],'llama:latest'), (['llama:latest'],'llama:latest'), (['bad','2'],available.default)]:
            with patch.object(JD_RAG,'list_chat_models', return_value=available), patch('builtins.input',side_effect=values), patch('builtins.print'):
                self.assertEqual(JD_RAG.choose_llm_model(self.settings), expected)
        with patch.object(JD_RAG,'list_chat_models', return_value=available), patch('builtins.input',side_effect=EOFError), patch('builtins.print'):
            self.assertEqual(JD_RAG.choose_llm_model(self.settings), available.default)

    def test_chat_returns_answer_and_sources(self):
        doc = NS(page_content='Evidence', metadata={'source':'/notes/deck.pptx','slide':8,'type':'slide_text'})
        db, llm, prompt = Mock(), Mock(), Mock()
        db.as_retriever.return_value.invoke.return_value = [doc]
        llm.invoke.return_value.content = 'Answer'
        session = ChatSession(db, self.settings, 'chosen:model', llm=llm, prompt=prompt)
        answer = session.ask('Question')
        self.assertEqual((answer.text, answer.model), ('Answer','chosen:model'))
        self.assertEqual(answer.sources[0].label, 'deck.pptx, slide 8')
        self.assertEqual(answer.sources[0].metadata, doc.metadata)
        prompt.invoke.assert_called_once_with({'question':'Question','context':'Source: deck.pptx slide 8\nContent type: slide_text\nEvidence'})
        db.as_retriever.assert_called_once_with(search_kwargs={'k':4})

    def test_empty_chat_does_not_invoke_model(self):
        db, llm = Mock(), Mock()
        db.as_retriever.return_value.invoke.return_value = []
        session = ChatSession(db, self.settings, 'chosen', llm=llm, prompt=Mock())
        self.assertFalse(session.ask('Question').sources)
        llm.invoke.assert_not_called()

    def test_source_dedup_keeps_all_context(self):
        doc = NS(page_content='Evidence', metadata={'source':'/notes/paper.pdf','page':3})
        context, sources = format_context([doc, doc])
        self.assertEqual(len(sources), 1)
        self.assertEqual(context.count('Evidence'), 2)
        self.assertEqual(sources[0].label, 'paper.pdf, page 3')


if __name__ == '__main__':
    unittest.main()
