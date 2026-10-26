# ESX RAG — Personal Retrieval-Augmented Generation Agent

A local, private RAG system that lets you ask questions against your own documents (PDF, TXT, MD) using Ollama LLMs and ChromaDB vector storage. Includes both a **command-line interface** and a **Streamlit web UI**.

## Project Structure

```
esx_rag/
├── rag_agent.py          # Backend RAG logic (CLI + importable functions)
├── streamlit_app.py      # Streamlit web UI
├── documents/            # Your document collections (one folder per collection)
│   └── esx_data/
│       └── 1150_Collective_Investment_Schemes_Operation_Directive_No_1150_2.pdf
├── chroma_data/          # ChromaDB persistent storage (created automatically)
├── .gitignore
└── README.md
```

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([download](https://ollama.com/download))

### 2. Pull the required models

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 3. Install Python dependencies

```powershell
# Create and activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install streamlit chromadb ollama pypdf
```

### 4. Add your documents

Place your files in a folder under `./documents/`. The folder name becomes your **collection name**:

```
documents/
└── my_collection/
    ├── document1.pdf
    ├── document2.txt
    └── document3.md
```

Supported formats: **PDF**, **TXT**, **MD**

### 5. Run the application

#### Option A — Web UI (Recommended)

```powershell
streamlit run streamlit_app.py
```

Open your browser to **http://localhost:8501**

#### Option B — Command Line

```powershell
python rag_agent.py
```

## Using the Web UI

1. **Launch**: `streamlit run streamlit_app.py`
2. **Sidebar**:
   - Check Ollama connection status (must show ✅ Connected)
   - View existing collections
   - Ingest a new collection by entering the folder name under `./documents/`
   - Adjust the number of retrieved chunks (1–10)
3. **Main chat**:
   - Select a collection from the dropdown
   - Type your question in the chat box
   - Read the answer with source citations (expand the "📚 Sources" section to see excerpts)

## Using the CLI

```powershell
python rag_agent.py
```

Interactive menu:
- **[I]ngest** — Index a document collection
- **[A]sk** — Ask a question against a collection
- **[L]ist** — Show all collections
- **[E]xit** — Quit

## Configuration

Edit the constants at the top of `rag_agent.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `CHAT_MODEL` | `qwen3:4b` | LLM used for generating answers |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vector search |
| `CHROMA_DB_PATH` | `./chroma_data` | Local vector database storage |
| `DOCUMENTS_PATH` | `./documents` | Folder containing document collections |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |

## Example Workflow

```powershell
# 1. Start Ollama (separate terminal)
ollama serve

# 2. Organize documents
mkdir documents\finance_regulations
copy *.pdf documents\finance_regulations\

# 3. Run the UI
streamlit run streamlit_app.py

# 4. In the UI sidebar:
#    - Enter "finance_regulations" as collection name
#    - Click " Ingest collection"
#    - Select the collection and start chatting
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to Ollama` | Make sure Ollama is running (`ollama serve`) and the models are pulled |
| `No collections found` | Ingest a collection first via the sidebar or CLI |
| `No relevant documents found` | Try rephrasing your question or increase the retrieved chunks in settings |
| Sources text not visible | Make sure you're using the latest `streamlit_app.py` (source boxes now have dark text) |

## License

This project is for personal/educational use.

---

**Built with:** Ollama • ChromaDB • Streamlit • Python