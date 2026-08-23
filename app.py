import os
from pathlib import Path
import streamlit as st
from groq import Groq
from rag_engine import RAGEngine

st.set_page_config(page_title="GenAI Lab 3: Document RAG Engine", page_icon="📚", layout="wide")

# 1. Safe Multi-Source API Authentication
api_key = None

# A. Try loading from local .env first
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.getenv("GROQ_API_KEY")
except Exception:
    pass

# B. Fallback to Streamlit Secrets (for Streamlit Cloud deployment)
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

if not api_key:
    st.error("❌ `GROQ_API_KEY` not found in `.env` or Streamlit Secrets.")
    st.info("💡 Ensure your `.env` file contains: `GROQ_API_KEY=gsk_your_actual_key`")
    st.stop()

# 2. Cached Engine Setup
@st.cache_resource(show_spinner="Loading FastEmbed model into memory...")
def init_rag_engine(api_key_str: str):
    client = Groq(api_key=api_key_str.strip())
    return RAGEngine(groq_client=client, model_name="openai/gpt-oss-20b")

rag = init_rag_engine(api_key)

if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 3. Sidebar: Document Ingestion & Parameters
with st.sidebar:
    st.header("📄 Ingest Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("⚡ Process & Index Document", use_container_width=True):
            with st.spinner("Extracting text and indexing into ChromaDB..."):
                if uploaded_file.name.endswith(".pdf"):
                    raw_text = rag.extract_text_from_pdf(uploaded_file)
                else:
                    raw_text = uploaded_file.read().decode("utf-8")
                
                num_chunks = rag.index_document(raw_text, uploaded_file.name)
                st.session_state.indexed_doc = uploaded_file.name
                st.success(f"Indexed **{num_chunks} chunks** successfully!")

    st.divider()
    st.header("⚙️ Search Configuration")
    top_k = st.slider("Top-K Semantic Chunks", min_value=1, max_value=6, value=3)
    
    if st.session_state.indexed_doc:
        st.info(f"📂 Active Document: **{st.session_state.indexed_doc}**")
    else:
        st.warning("⚠️ No document indexed yet.")
        
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 4. Main Chat Interface
st.title("📚 GenAI Lab 3: Document RAG Engine")
st.caption("Local Vector Search (ChromaDB + FastEmbed) + Grounded LLM Context (Groq)")

# Display conversation history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if "sources" in chat and chat["sources"]:
            with st.expander("🔍 View Retrieved Context Chunks"):
                for src in chat["sources"]:
                    st.markdown(f"**Chunk #{src['chunk_id']}** (Source: `{src['source']}`)")
                    st.code(src["content"], language="text")

# Chat input
if query := st.chat_input("Ask a question about your uploaded document..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB and synthesizing answer..."):
            result = rag.answer_query(query, top_k=top_k)
            st.markdown(result["answer"])
            
            if result["sources"]:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for src in result["sources"]:
                        st.markdown(f"**Chunk #{src['chunk_id']}** (Source: `{src['source']}`)")
                        st.code(src["content"], language="text")

    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": result["answer"],
        "sources": result["sources"]
    })