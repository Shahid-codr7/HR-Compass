"""
HR Policy RAG Assistant
=======================
A Retrieval-Augmented Generation (RAG) pipeline that answers employee
questions strictly from HR policy documents (PDF/TXT).

Pipeline (same approach as the YouTube RAG chatbot):
    Load -> Split -> Embed -> Store (FAISS) -> Retrieve -> Augment -> Generate

Setup:
    pip install langchain langchain-core langchain-community langchain-google-genai faiss-cpu pypdf

    Set your Gemini API key before running:
        export GOOGLE_API_KEY="your-api-key-here"
"""

import os
import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# Folder containing the HR policy documents (.pdf and/or .txt files)
HR_DOCS_PATH = "./hr_policies"


# --------------------------------------------------------------------------- #
# Step 1 - Ingestion: load HR policy documents from disk
# --------------------------------------------------------------------------- #
def load_hr_documents(folder_path: str) -> list:
    """Load all .pdf and .txt files from `folder_path` into LangChain Documents."""
    documents = []

    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if file_path.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(file_path).load())
        elif file_path.lower().endswith(".txt"):
            documents.extend(TextLoader(file_path, encoding="utf-8").load())

    return documents


# --------------------------------------------------------------------------- #
# Step 2 - Chunking: split documents into overlapping chunks
# --------------------------------------------------------------------------- #
def split_documents(documents: list) -> list:
    """Split documents into chunks (chunk_size=2000, chunk_overlap=700)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=700)
    return text_splitter.split_documents(documents)


# --------------------------------------------------------------------------- #
# Step 3 - Embedding & Vector Store: embed chunks and store them in FAISS
# --------------------------------------------------------------------------- #
def build_vector_store(chunks: list) -> FAISS:
    """Embed document chunks with Gemini embeddings and store them in FAISS."""
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.from_documents(chunks, embed_model)


# --------------------------------------------------------------------------- #
# Step 4 - Prompt Template: grounded, hallucination-resistant prompt
# --------------------------------------------------------------------------- #
prompt = PromptTemplate(
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
# Step 5 - format_docs: join retrieved chunks into a single context string
# --------------------------------------------------------------------------- #
def format_docs(docs: list) -> str:
    """Join retrieved chunk contents with double newlines for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------------------------------------------------------- #
# Step 6 - Chain Construction (LCEL): retriever -> prompt -> LLM -> parser
# --------------------------------------------------------------------------- #
def build_rag_chain(vector_store: FAISS):
    """Construct the retrieval-augmented generation chain using LCEL."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    return parallel_chain | prompt | chat_model | parser


# --------------------------------------------------------------------------- #
# Main - build the pipeline once, then answer questions interactively
# --------------------------------------------------------------------------- #
def main():
    print("Loading HR policy documents...")
    documents = load_hr_documents(HR_DOCS_PATH)
    if not documents:
        raise FileNotFoundError(f"No .pdf or .txt files found in '{HR_DOCS_PATH}'.")

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks and building FAISS vector store...")
    vector_store = build_vector_store(chunks)

    chain = build_rag_chain(vector_store)

    print("\nHR Policy Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        print(f"\nAnswer: {chain.invoke(question)}\n")


if __name__ == "__main__":
    main()
