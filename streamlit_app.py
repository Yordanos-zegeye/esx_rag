"""
Streamlit UI for the Personal RAG Agent.

Run with:
    streamlit run streamlit_app.py

Requires Ollama to be running at http://localhost:11434
and the models qwen3:4b (chat) and nomic-embed-text (embeddings) pulled.
"""

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_agent import (  # noqa: E402
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    check_ollama_connection,
    generate_answer,
    ingest_collection,
    list_collections,
    ollama_client,
    retrieve,
)

st.set_page_config(
    page_title="Personal RAG Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9em;
        color: #212529;
    }
    .source-box strong {
        color: #0056b3;
    }
    .source-box br {
        display: block;
        margin: 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – settings & collection management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔧 RAG Agent")
    st.caption("Personal Retrieval-Augmented Generation")

    st.divider()
    st.subheader("Ollama Status")

    if "ollama_ok" not in st.session_state:
        st.session_state.ollama_ok = check_ollama_connection()

    if st.session_state.ollama_ok:
        st.success(f"✅ Connected to Ollama at {OLLAMA_BASE_URL}")
        st.write(f"**Chat model:** `{CHAT_MODEL}`")
        st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    else:
        st.error(f"❌ Cannot reach Ollama at {OLLAMA_BASE_URL}")
        st.info(
            "Please make sure Ollama is running:\n"
            "1. Download from https://ollama.com/download\n"
            "2. Pull the models:\n   `ollama pull qwen3:4b`\n   `ollama pull nomic-embed-text`"
        )
        if st.button("Re-check connection"):
            st.session_state.ollama_ok = check_ollama_connection()
            st.rerun()

    st.divider()
    st.subheader("Collections")

    if st.button("🔄 Refresh collections"):
        try:
            st.session_state.collections = list_collections()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to list collections: {exc}")

    if "collections" not in st.session_state:
        try:
            st.session_state.collections = list_collections()
        except Exception as exc:  # noqa: BLE001
            st.session_state.collections = []
            st.error(f"Failed to list collections: {exc}")

    if st.session_state.collections:
        st.write("Available collections:")
        for col in st.session_state.collections:
            st.write(f"• {col}")
    else:
        st.write("No collections found.")

    st.divider()
    st.subheader("Ingest New Collection")

    new_collection = st.text_input(
        "Collection name",
        placeholder="e.g. esx_data",
        help="Folder name under ./documents containing your PDF/TXT/MD files",
    )

    if st.button("📥 Ingest collection", type="primary"):
        if not new_collection:
            st.warning("Please enter a collection name.")
        else:
            with st.spinner(f"Ingesting '{new_collection}'..."):
                try:
                    ingest_collection(new_collection)
                    st.session_state.collections = list_collections()
                    st.success(f"Collection '{new_collection}' ingested successfully!")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Settings")
    n_results = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="How many document chunks to retrieve per question.",
    )
    st.session_state.n_results = n_results

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("💬 Personal RAG Chat")
st.caption("Ask questions about your ingested documents. Answers are grounded in your files.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source:</strong> {src['source']}<br>
                            <strong>Distance:</strong> {src['distance']:.4f}<br>
                            <strong>Excerpt:</strong> {src['text'][:300]}{'...' if len(src['text']) > 300 else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

if st.session_state.collections:
    selected_collection = st.selectbox(
        "Select collection to search",
        options=st.session_state.collections,
        index=0,
    )
else:
    st.warning("No collections available. Please ingest a collection first using the sidebar.")
    selected_collection = None

if prompt := st.chat_input("Ask a question about your documents..."):
    if not selected_collection:
        st.error("Please select or ingest a collection before asking questions.")
    elif not st.session_state.get("ollama_ok", False):
        st.error("Ollama is not reachable. Please start Ollama and try again.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks = retrieve(
                        selected_collection, prompt, n_results=st.session_state.get("n_results", 3)
                    )

                    if not chunks:
                        answer = "I don't know — no relevant documents were found in the selected collection."
                        sources = []
                    else:
                        answer = generate_answer(prompt, chunks)
                        sources = chunks

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <strong>Source:</strong> {src['source']}<br>
                                        <strong>Distance:</strong> {src['distance']:.4f}<br>
                                        <strong>Excerpt:</strong> {src['text'][:300]}{'...' if len(src['text']) > 300 else ''}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:  # noqa: BLE001
                    error_msg = f"❌ Error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )
else:
    if not st.session_state.collections:
        st.info(
            "👋 Welcome! To get started:\n\n"
            "1. Place your PDF, TXT, or MD files in a folder under `./documents/`\n"
            "2. Use the sidebar to ingest that folder as a collection\n"
            "3. Select the collection and start asking questions!"
        )
