"""
HR Policy RAG Assistant - Streamlit App
========================================
Upload HR policy documents (PDF/TXT), build a FAISS-backed knowledge base
using Google Gemini embeddings, and chat with an LLM that answers strictly
from the uploaded policy content.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

The Google API key is NEVER entered or displayed in the UI. It is read
silently from Streamlit secrets or an environment variable:
    - Local: create .streamlit/secrets.toml with GOOGLE_API_KEY = "your-key"
             (or set the GOOGLE_API_KEY environment variable)
    - Streamlit Community Cloud: add GOOGLE_API_KEY under the app's
      "Settings -> Secrets"

Deploy on Streamlit Community Cloud:
    1. Push app.py + requirements.txt to a GitHub repo.
    2. Create a new app on share.streamlit.io pointing to app.py.
    3. In the app's "Secrets" settings, add: GOOGLE_API_KEY = "your-api-key"
"""

import os
import asyncio
import tempfile

# --------------------------------------------------------------------------- #
# Fix: "There is no current event loop in thread 'ScriptRunner.scriptThread'"
#
# Root cause: Streamlit runs the script in a background worker thread, which
# (in Python 3.10+) has no asyncio event loop by default. The google-generativeai
# SDK used by GoogleGenerativeAIEmbeddings / ChatGoogleGenerativeAI calls
# asyncio.get_event_loop() internally (e.g. when building its gRPC channel),
# which raises RuntimeError if no loop exists in the current thread.
# Creating and registering a loop for this thread before any Google GenAI
# client is instantiated resolves it.
# --------------------------------------------------------------------------- #
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st

# --------------------------------------------------------------------------- #
# Fix: "ModuleNotFoundError: No module named 'langchain_community'" (and similar)
#
# Root cause: this app's dependencies (langchain, langchain-community,
# langchain-google-genai, faiss-cpu, pypdf, ...) aren't installed in the
# Python interpreter that is actually running `streamlit run app.py`.
# This commonly happens when a different interpreter/venv is selected by
# your editor/runner than the one where you ran `pip install -r requirements.txt`.
#
# Instead of letting this crash with a raw traceback, show an actionable
# message telling the user exactly how to fix it.
# --------------------------------------------------------------------------- #
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
except ModuleNotFoundError as exc:
    st.set_page_config(page_title="HR Policy Assistant", page_icon="📄")
    st.error(
        f"**Missing dependency: `{exc.name}`**\n\n"
        "The packages this app needs aren't installed in the Python "
        "environment that's running this app.\n\n"
        "**Fix:**\n"
        "1. Open a terminal in this project's folder.\n"
        "2. Activate the environment you intend to use "
        "(e.g. `conda activate lgchain`).\n"
        "3. Run:\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "```\n"
        "4. Re-run `streamlit run app.py` **from that same terminal/environment**.\n\n"
        "If you're using an IDE/runner (e.g. a 'Code Playground' extension), "
        "make sure it's configured to use the same Python interpreter where "
        "the packages were installed."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Fixed pipeline configuration (per project spec)
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 700
RETRIEVER_K = 2
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"],
)


# --------------------------------------------------------------------------- #
# API key - loaded silently, never shown or echoed back in the UI
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    """Read the Google API key from Streamlit secrets or the environment.

    The key is intentionally never rendered in a widget, so it can't be
    displayed, copied from the page, or seen via browser dev tools.
    """
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_API_KEY", "")


# --------------------------------------------------------------------------- #
# Pipeline functions
# --------------------------------------------------------------------------- #
def load_uploaded_documents(uploaded_files) -> list:
    """Convert uploaded PDF/TXT files into a list of LangChain Documents."""
    documents = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        if suffix == ".pdf":
            # PyPDFLoader needs a real file path, so write to a temp file first.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                file_docs = PyPDFLoader(tmp_path).load()
            finally:
                os.remove(tmp_path)
        elif suffix == ".txt":
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            file_docs = [Document(page_content=text)]
        else:
            continue

        for doc in file_docs:
            doc.metadata["source"] = uploaded_file.name
        documents.extend(file_docs)

    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list) -> FAISS:
    """Embed document chunks with Gemini embeddings and store them in FAISS."""
    embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embed_model)


def format_docs(docs: list) -> str:
    """Join retrieved chunk contents with double newlines for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: FAISS):
    """Construct the retrieval -> prompt -> LLM -> parser chain using LCEL."""
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )
    chat_model = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    parser = StrOutputParser()

    # Retrieve once, then fan out into (a) formatted context for the prompt
    # and (b) the raw source documents, kept around to show in the UI.
    retrieval_step = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "source_documents": retriever,
        }
    )

    generation_step = RunnableParallel(
        {
            "answer": PROMPT | chat_model | parser,
            "source_documents": RunnableLambda(lambda x: x["source_documents"]),
        }
    )

    return retrieval_step | generation_step


# --------------------------------------------------------------------------- #
# Helper: render a "Sources" expander for a list of retrieved documents
# --------------------------------------------------------------------------- #
def render_sources(source_documents: list) -> None:
    with st.expander("📚 Sources"):
        for i, doc in enumerate(source_documents, start=1):
            source = doc.metadata.get("source", "uploaded document")
            page = doc.metadata.get("page")
            label = source + (f" (page {page + 1})" if page is not None else "")
            preview = doc.page_content[:500]
            if len(doc.page_content) > 500:
                preview += "..."
            st.markdown(f"**{i}. {label}**")
            st.text(preview)


# --------------------------------------------------------------------------- #
# Streamlit page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="HR Policy Assistant", page_icon="📄", layout="wide")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("rag_chain", None)
st.session_state.setdefault("num_chunks", 0)

# --------------------------------------------------------------------------- #
# Sidebar - configuration, document upload, knowledge base controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    # API key is loaded silently from secrets/env - never shown in the UI.
    api_key = get_api_key()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API key configured", icon="🔒")
    else:
        st.error("API key not configured", icon="🔒")
        st.caption(
            "Set `GOOGLE_API_KEY` in `.streamlit/secrets.toml` "
            "(local) or in the app's Secrets settings (Streamlit Cloud)."
        )

    st.divider()
    st.header("📁 HR Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT policy files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🔄 Build Knowledge Base", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Google API key is not configured. See sidebar message above.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF or TXT file.")
        else:
            try:
                with st.spinner("Reading documents..."):
                    documents = load_uploaded_documents(uploaded_files)
                with st.spinner("Splitting into chunks..."):
                    chunks = split_documents(documents)
                with st.spinner("Generating embeddings & building FAISS index..."):
                    vector_store = build_vector_store(chunks)
                    st.session_state.rag_chain = build_rag_chain(vector_store)
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.messages = []
                st.success(f"Knowledge base ready - {len(chunks)} chunks indexed.")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")

    if st.session_state.num_chunks:
        st.caption(f"✅ Indexed {st.session_state.num_chunks} chunks")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Pipeline details"):
        st.markdown(
            f"""
- **Chunking:** size `{CHUNK_SIZE}`, overlap `{CHUNK_OVERLAP}`
- **Embeddings:** `{EMBEDDING_MODEL}`
- **Vector store:** FAISS (similarity search, k=`{RETRIEVER_K}`)
- **LLM:** `{LLM_MODEL}` (temperature `{LLM_TEMPERATURE}`)
"""
        )

# --------------------------------------------------------------------------- #
# Main area - chat interface
# --------------------------------------------------------------------------- #
st.title("📄 HR Policy RAG Assistant")
st.caption(
    "Ask questions about your company's HR policies. "
    "Answers are grounded strictly in the uploaded documents."
)

if st.session_state.rag_chain is None:
    st.info("👈 Upload your HR policy documents and click **Build Knowledge Base** to get started.")
else:
    # Replay chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask about leave policy, benefits, code of conduct, etc.")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke(question)
                    answer = result["answer"]
                    sources = result["source_documents"]
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    sources = []

            st.markdown(answer)
            if sources:
                render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )