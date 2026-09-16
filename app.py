import os
import tempfile
import streamlit as st
from src.rag_engine import RAGEngine
from src.config import settings

st.set_page_config(page_title="Enterprise Local RAG Agent", page_icon="🦙", layout="wide")

st.title("🦙 Enterprise Local Knowledge Base (Ollama RAG)")
st.caption("100% lokální vyhledávání v dokumentech bez odesílání dat do cloudu")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Konfigurace")
    provider = st.selectbox("Provider LLM & Embeddings", ["Ollama", "Azure"], index=0)

    if provider == "Ollama":
        ollama_url = st.text_input("Ollama URL", value=settings.OLLAMA_BASE_URL)
        settings.OLLAMA_BASE_URL = ollama_url
        st.info("Běží na lokálním Ollamě: `llama3.2` + `nomic-embed-text`")
    elif provider == "Azure":
        st.subheader("Azure OpenAI Credentials")
        settings.AZURE_OPENAI_ENDPOINT = st.text_input("Azure Endpoint", value=settings.AZURE_OPENAI_ENDPOINT)
        settings.AZURE_OPENAI_API_KEY = st.text_input("Azure API Key", type="password", value=settings.AZURE_OPENAI_API_KEY)
        settings.AZURE_CHAT_DEPLOYMENT = st.text_input("Chat Deployment", value=settings.AZURE_CHAT_DEPLOYMENT)
        settings.AZURE_EMBEDDING_DEPLOYMENT = st.text_input("Embedding Deployment", value=settings.AZURE_EMBEDDING_DEPLOYMENT)

    st.divider()
    st.subheader("📄 Nahrajte dokumentaci")
    uploaded_files = st.file_uploader("Nahrát PDF soubory", type=["pdf"], accept_multiple_files=True)

# Správná invalidace cache podle nastavených parametrů
@st.cache_resource
def get_rag_engine(provider_name: str, **config_key) -> RAGEngine:
    return RAGEngine(provider=provider_name)

if provider == "Ollama":
    rag = get_rag_engine("ollama", url=settings.OLLAMA_BASE_URL)
else:
    rag = get_rag_engine(
        "azure", 
        endpoint=settings.AZURE_OPENAI_ENDPOINT, 
        key=settings.AZURE_OPENAI_API_KEY,
        chat_dep=settings.AZURE_CHAT_DEPLOYMENT,
        embed_dep=settings.AZURE_EMBEDDING_DEPLOYMENT
    )

if uploaded_files and st.sidebar.button("Indexovat dokumenty do ChromaDB", type="primary"):
    with st.spinner("Zpracovávám PDF přes embeddings..."):
        try:
            total_chunks = 0
            for uploaded_file in uploaded_files:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    chunks_count = rag.process_pdf(tmp_path, uploaded_file.name)
                    total_chunks += chunks_count
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.sidebar.success(f"Indexováno {len(uploaded_files)} souborů ({total_chunks} bloků)!")
        except Exception as e:
            st.sidebar.error(f"Chyba při indexaci: {e}")

# Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Citace a zdroje"):
                for src in msg["sources"]:
                    st.write(f"• **{src['file']}** (Strana {src['page']})")
                    st.caption(f'"{src["excerpt"]}..."')

# Chat Input
if user_input := st.chat_input("Ptejte se na obsah nahraných PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            stream, source_docs = rag.query(user_input)

            response_placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                content_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += content_text
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            sources_metadata = []
            with st.expander("📚 Citace a zdroje"):
                for doc in source_docs:
                    meta = doc.metadata
                    excerpt = doc.page_content[:200].replace("\n", " ")
                    st.write(f"• **{meta.get('source_file', 'Soubor')}** (Strana {meta.get('page', '?')})")
                    st.caption(f'"{excerpt}..."')
                    
                    sources_metadata.append({
                        "file": meta.get("source_file", "Neznámý"),
                        "page": meta.get("page", "?"),
                        "excerpt": excerpt
                    })

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_metadata
            })

        except Exception as e:
            st.error(f"Chyba: {str(e)}")