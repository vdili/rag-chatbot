# RAG Chatbot (PDF Knowledge Base)

A small RAG chatbot that answers questions using a set of uploaded PDF documents.
It retrieves relevant passages, generates an answer based only on those passages, and shows citations (document + page + chunk).
If the information is not in the documents, it returns: **"Not found in the uploaded documents."**

## What it does
- Ingest PDFs from `data/raw/`
- Clean text and split into chunks
- Create embeddings and build a FAISS vector index
- Retrieve Top-K relevant chunks (+ BM25 reranking)
- Generate a short answer grounded in retrieved chunks (OpenAI)
- Show citations for transparency

## Project structure
rag-chatbot/
app/streamlit_app.py
rag/ingest.py
rag/generation.py
data/raw/
requirements.txt
.gitignore
README.md


## Run ingestion
./.venv/Scripts/python.exe rag/ingest.py

## Run the app
./.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

## Open in browser
http://localhost:8501




