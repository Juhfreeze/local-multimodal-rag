# JD_RAG

A local Retrieval-Augmented Generation (RAG) system for macOS that lets you ask questions about your own documents using locally running AI models.

JD_RAG combines:

- **Qwen3 4B Instruct** for answering questions
- **Nomic Embed Text** for document and query embeddings
- **Granite 3.2 Vision** for visual document understanding
- **Docling** for document extraction and OCR
- **Chroma** for local vector search
- **Ollama** for local model serving

Your documents and vector database remain on your computer during normal use.

> **Platform support:** JD_RAG is currently tested and documented for **macOS only**. Windows and Linux support may be added after those platforms have been tested.

---

## Features

- Local document question answering
- Runs local models through Ollama
- Supports PDF, PowerPoint, Word, Markdown, and plain-text files
- OCR and structured document extraction with Docling
- Visual analysis of charts, graphs, diagrams, and image-based equations with Granite Vision
- Chroma vector search
- Incremental indexing
- Automatically detects new, modified, and deleted documents
- Avoids rebuilding the entire database when only a few files change
- Preserves source filenames and page/slide references where available
- `/update` command for rescanning documents while the program is running
- Granite Vision is unloaded after indexing to reduce memory usage during normal chat
- No cloud API is required for normal operation after required models and dependencies are installed

---

## 1. Prerequisites

You need:

- macOS
- Homebrew
- Python 3.13
- Ollama
- LibreOffice for full PowerPoint slide rendering
- Enough free disk space for the Ollama models, Python packages, and your local document index

Internet access is required during initial setup to download packages and models.

### Install Homebrew

Check whether Homebrew is already installed:

```bash
brew --version
```

If a version is displayed, continue to the Python section.

If Terminal says `command not found`, install Homebrew using the official installer:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the installer's prompts and complete the **Next steps** it prints at the end.

Then verify:

```bash
brew --version
```

Homebrew: https://brew.sh/

---

## 2. Install Python 3.13

```bash
brew install python@3.13
python3.13 --version
```

If `python3.13 --version` already works, you can skip the install command.

JD_RAG uses a Python virtual environment, so its packages remain isolated from your system Python installation.

---

## 3. Install Ollama

The `ollama` package in `requirements.txt` is the Python client. You also need the Ollama application itself to run the models.

Install Ollama with Homebrew:

```bash
brew install --cask ollama-app
open -a Ollama
```

Alternatively, install Ollama from:

https://ollama.com/download/mac

Use either installation method, not both.

Verify:

```bash
ollama --version
```

Leave the Ollama application running while using JD_RAG.

---

## 4. Download the required Ollama models

Check which models are already installed:

```bash
ollama list
```

Pull any that are missing:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
ollama pull granite3.2-vision
```

### Model roles

| Model | Purpose |
| --- | --- |
| `qwen3:4b-instruct` | Answers questions using retrieved document context |
| `nomic-embed-text` | Creates embeddings for documents and search queries |
| `granite3.2-vision` | Interprets charts, diagrams, equations, and other visual content during indexing |

Docling is **not** an Ollama model. It is installed as a Python package and may download additional processing models during the first indexing run.

---

## 5. Install LibreOffice

LibreOffice is used to render complete PowerPoint slides so Granite Vision can inspect native charts, diagrams, and equation objects.

```bash
brew install --cask libreoffice
```

If LibreOffice is already installed, skip this step.

Without LibreOffice, JD_RAG can still extract PowerPoint text, tables, chart data, and embedded images, but some native slide visuals may not be available to Granite Vision.

---

## 6. Create the project folder

Create the project directory and the folder that will contain your documents:

```bash
mkdir -p ~/JD_RAG/my_notes
cd ~/JD_RAG
```

Place these repository files directly inside `~/JD_RAG`:

```text
JD_RAG.py
requirements.txt
README.md
LICENSE
.gitignore
```

After setup and indexing, your local directory will look similar to:

```text
JD_RAG/
├── JD_RAG.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── my_notes/           # Your documents
├── .venv/              # Local Python environment
├── chroma_db/          # Local Chroma database
└── indexed_files.json  # Local indexing manifest
```

The final four local/generated items should not be committed to the repository.

---

## 7. Create the Python virtual environment

From the project directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

After activation, your Terminal prompt should begin with something similar to:

```text
(.venv)
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Whenever you open a new Terminal session, activate the environment again before running JD_RAG:

```bash
cd ~/JD_RAG
source .venv/bin/activate
```

---

## 8. Add your documents

Place the documents you want to search inside:

```text
my_notes/
```

Subfolders are supported.

### Supported file types

| Extension | Type |
| --- | --- |
| `.txt` | Plain text |
| `.md` | Markdown |
| `.pdf` | PDF |
| `.pptx` | PowerPoint |
| `.docx` | Microsoft Word |

Older `.ppt` and `.doc` files and standalone image files are not currently supported as direct inputs.

Images embedded inside supported documents can be analyzed during indexing.

---

## 9. Check and run JD_RAG

First check the Python syntax:

```bash
python -m py_compile JD_RAG.py
```

If nothing is printed, the syntax check passed.

Then start the program:

```bash
python JD_RAG.py
```

On the first run you should see:

```text
Your database is empty.
Index your documents now? (y/n):
```

Enter:

```text
y
```

JD_RAG will process the documents in `my_notes`, create embeddings, and build the local Chroma database.

The first indexing run can take longer because Docling may need to initialize or download processing models, and Granite Vision may analyze visual content.

---

## 10. Normal startup

After the database exists, JD_RAG asks:

```text
Check my_notes for new, modified or deleted files? (y/n):
```

Choose:

- **`y`** if you added, changed, renamed, or deleted documents.
- **`n`** if you only want to chat with the existing index.

Choosing `n` skips document processing and starts the RAG faster.

---

## 11. Updating documents

JD_RAG tracks document changes using a local manifest.

It detects:

- New files
- Modified files
- Deleted files

Unchanged files are skipped.

You can update in either of two ways:

1. Restart JD_RAG and answer `y` when asked to check for changes.
2. Type this while the RAG is running:

```text
/update
```

If indexing a particular file fails, that file will be retried during a future update.

> `/update` detects changes to your documents. It does not automatically detect major changes to the extraction pipeline, chunking logic, or embedding model. Those changes may require rebuilding `chroma_db` and `indexed_files.json`.

---

## 12. Using the RAG

After indexing completes, ask questions at the prompt:

```text
You:
```

Example:

```text
You: What does the document say about the CAPM?
```

JD_RAG retrieves relevant chunks from Chroma and passes them to Qwen.

Retrieved filenames and page or slide numbers are displayed when available.

### Chat commands

| Command | Action |
| --- | --- |
| `/update` | Rescan and synchronize `my_notes` |
| `quit` | Exit |
| `exit` | Exit |

---

## 13. Useful commands

See installed Ollama models:

```bash
ollama list
```

See models currently loaded into memory:

```bash
ollama ps
```

Start JD_RAG from a new Terminal session:

```bash
cd ~/JD_RAG
source .venv/bin/activate
python JD_RAG.py
```

Leave the Python virtual environment after exiting:

```bash
deactivate
```

If JD_RAG cannot reach Ollama, make sure the Ollama application is running.

---

## 14. Repository contents

A clean repository should contain:

```text
JD_RAG.py
requirements.txt
README.md
LICENSE
.gitignore
```

Do **not** commit:

```text
.venv/
my_notes/
chroma_db/
indexed_files.json
__pycache__/
.env
.DS_Store
```

Your `my_notes` folder may contain private or copyrighted documents, so it should remain local unless you intentionally choose to share those files.

A recommended `.gitignore` is:

```gitignore
# Python
.venv/
venv/
__pycache__/
*.pyc
*.pyo

# JD_RAG local data
my_notes/
chroma_db/
indexed_files.json

# Environment variables / secrets
.env
*.env

# macOS
.DS_Store

# Local archives
*.zip
```

---

## 15. Privacy and local operation

JD_RAG is designed to use local models through Ollama and a local Chroma database.

After the required software, packages, and models are installed, normal RAG use does not require sending your document contents to a cloud LLM API.

Docling or other dependencies may require internet access during initial installation or first-time model downloads.

Always review third-party tools and model licenses before using the project with sensitive, regulated, or confidential information.

---

## 16. Third-party software and models

JD_RAG depends on third-party projects including Ollama, Docling, LangChain, Chroma, Qwen, Granite, Nomic Embed Text, LibreOffice, PyMuPDF, Pillow, and python-pptx.

Those projects and models remain subject to their own respective licenses and terms.

The AGPL-3.0 license in this repository applies to the original JD_RAG project code; it does not replace the licenses of third-party dependencies or models.

---

## 17. Platform support

JD_RAG is currently **tested on macOS**.

Windows and Linux may work with modifications, but they are not currently documented as supported platforms because those environments have not yet been fully tested.

Cross-platform support and installation instructions may be added in a future release.

---

## License

JD_RAG is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You may use, study, modify, and redistribute the software, including commercially, subject to the terms of the AGPL-3.0.

If you modify covered software and make it available to users over a network, the AGPL includes source-code availability requirements described in the license.

See the [`LICENSE`](LICENSE) file for the complete license terms.

Third-party libraries, applications, and AI models used by JD_RAG remain subject to their own licenses.
