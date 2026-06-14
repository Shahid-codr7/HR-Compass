# HR Policy RAG Assistant (Streamlit)

A Retrieval-Augmented Generation chatbot that answers employee questions
strictly from your uploaded HR policy documents (PDF/TXT), using LangChain,
Google Gemini, and FAISS.

## Pipeline

1. **Upload** PDF/TXT HR policy files via the sidebar.
2. **Chunk** text with `RecursiveCharacterTextSplitter` (`chunk_size=2000`, `chunk_overlap=700`).
3. **Embed** chunks with `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`).
4. **Store** embeddings in a `FAISS` vector index.
5. **Retrieve** top `k=2` chunks via similarity search.
6. **Generate** a grounded answer with `ChatGoogleGenerativeAI` (`gemini-2.5-flash`, `temperature=0.2`),
   using an LCEL chain (`RunnableParallel` + `RunnablePassthrough` + prompt + `StrOutputParser`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Enter your Google API key in the sidebar (get one from
[Google AI Studio](https://aistudio.google.com/app/apikey)), upload your HR
policy files, click **Build Knowledge Base**, and start chatting.

## Deploy on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app,
   pointing it at `app.py` in your repo.
3. (Optional) In the app's **Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```
   This pre-fills the API key so users don't have to enter their own.
   If you skip this, each user can paste their own key in the sidebar.
4. Deploy. Upload your HR policy documents in the running app and click
   **Build Knowledge Base**.

## Notes

- The knowledge base is rebuilt in-memory each session (or when you click
  **Build Knowledge Base**) — it is not persisted to disk, so it must be
  rebuilt after the app restarts or the session ends.
- Answers are restricted to the uploaded documents; if the context doesn't
  contain the answer, the assistant says it doesn't know.
- Each assistant response includes a **Sources** expander showing which
  document chunks were used to generate the answer.
