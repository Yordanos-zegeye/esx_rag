import os
import sys
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils.embedding_functions.ollama_embedding_function import (OllamaEmbeddingFunction)
from ollama import Client
from pypdf import PdfReader

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b"
CHROMA_DB_PATH = "./chroma_data"
DOCUMENTS_PATH = "./documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

ollama_client = Client(host=OLLAMA_BASE_URL)

ollama_ef = OllamaEmbeddingFunction(
    url=OLLAMA_BASE_URL, model_name=EMBEDDING_MODEL, timeout=120
)


def check_ollama_connection() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        ollama_client.list()
        return True
    except Exception as e:
        print(f"Warning: Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")
        print("Please make sure Ollama is downloaded and running (https://ollama.com/download)")
        return False

chroma_client = PersistentClient(path=CHROMA_DB_PATH)


# document loading
def load_text_file(file_path: str) -> str:
    with open(file_path,'r',encoding = 'utf-8') as f:
        return f.read()
def load_pdf_file(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_document(file_path:str) -> str:
    path = Path(file_path)
    if path.suffix.lower() ==".pdf":
        return load_pdf_file(file_path)
    elif path.suffix.lower() in [".txt", ".md"]:
        return load_text_file(file_path)
    else:
        print(f"Unsupported file type: {path.suffix}")
        return ""

# chunking using sliding window
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def ingest_collection(collection_name: str) -> None:
    collection_dir = Path(DOCUMENTS_PATH) / collection_name
    if not collection_dir.exists():
        print(f"Directory {collection_dir} does not exist.")
        return
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=ollama_ef)

    files = list(collection_dir.iterdir())
    if not files:
        print(f"No files found in {collection_dir}.")
        return

    all_chunks = []
    all_metadatas = []
    all_ids = []
    for file_path in files:
        if file_path.is_file():
            print(f"Processing file: {file_path}")
            text = load_document(str(file_path))
            if not text:
                continue
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_path.stem}_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({"source": str(file_path), "chunk_index": i, "collection": collection_name})
                all_ids.append(doc_id)
    if all_chunks:
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Added {len(all_chunks)} chunks to collection '{collection_name}'.")
    else:
        print(f"No valid chunks found in collection '{collection_name}'.")

#Retrival
def retrieve(collection_name: str, query:str,n_results: int = 3) -> list[dict]:
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef,
    )
    results = collection.query(
        query_texts = [query],
        n_results = n_results,

    )

    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['documents'] else[]
    distances = results['distances'][0] if results['documents'] else []

    retrieved = []
    for doc,meta,dist in zip(documents, metadatas, distances):
        retrieved.append({
            'text':doc,
            'source':meta.get('source','unknown'),
            'distance':dist,
        })
    return retrieved

# Generation
def generate_answer(query: str, context_chunks: list[dict]) -> str:
    context_text = '\n\n'.join(f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks)
    system_prompt = f"You are a helpful assistant. Use the following context to answer the question. If the answer is not contained within the context, say 'I don't know'. Cite the sources of your information in your answer."
    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

    response = ollama_client.chat(
        model = CHAT_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    return response.message.content

# RAG Pipeline
def ask(collection_name: str, query: str) ->None:
    print(f"\nSearcihng '{collection_name}' for: {query}")
    print("-" * 50)

    chunks = retrieve(collection_name, query)

    if not chunks:
        print("No relevant documents found.")
        return
    answer = generate_answer(query, chunks)

    print(f"\nAnswer:\n {answer}")
    print("\nSources:")
    for chunk in chunks:
        print(f"- {chunk['source']} (distance: {chunk['distance']:.4f})")

def list_collections() -> list[str]:
    collections = chroma_client.list_collections()
    return [col.name for col in collections]

def main() -> None:
    print("Personal RAG Agent")
    print(f"Chat Model: {CHAT_MODEL}, Embedding Model: {EMBEDDING_MODEL}")
    print()

    while True:
        print("\nOptions:")
        print("[I]ngest a collection")
        print("[A]sk a question")
        print("[L]ist collections")
        print("[E]xit")
        choice = input("Enter your choice: ").strip().lower()

        if choice == "i":
            collection_name = input("Enter the collection name to ingest: ").strip()
            ingest_collection(collection_name)
        elif choice == "a":
            collection_name = input("Enter the collection name to search: ").strip()
            query = input("Enter your question: ").strip()
            ask(collection_name, query)
        elif choice == "l":
            collections = list_collections()
            print("Available Collections:")
            for col in collections:
                print(f"- {col}")
        elif choice == "e":
            print(" Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
    
