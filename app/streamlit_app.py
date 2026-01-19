import re
import json
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from rag.generation import generate_answer

INDEX_DIR = Path("data/index")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(s: str):
    return re.findall(r"[a-zA-Z0-9]+", s.lower())


@st.cache_resource
def load_everything():
    model = SentenceTransformer(MODEL)
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

    chunks = []
    with (INDEX_DIR / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return model, index, chunks


st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")

model, index, chunks = load_everything()

with st.sidebar:
    st.write("Chunks loaded:", len(chunks))
    st.write("Index size:", index.ntotal)

    top_k = st.slider("Top-K passages", 3, 12, 6)
    dense_k = st.slider("Dense candidates (FAISS)", 10, 80, 30)
    threshold = st.slider("Relevance threshold (dense)", 0.05, 0.70, 0.15)

q = st.text_input("Ask a question", placeholder="Try: exposure analytics insurance")

if q:
    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    scores, ids = index.search(q_emb, dense_k)

    candidates = []
    for s, idx in zip(scores[0], ids[0]):
        if int(idx) < 0:
            continue

        ch = chunks[int(idx)]
        candidates.append(
            {
                "dense_score": float(s),
                "chunk_id": ch["chunk_id"],
                "doc_name": ch["doc_name"],
                "page": ch["page"],
                "text": ch["text"],
            }
        )

    if not candidates or max(c["dense_score"] for c in candidates) < threshold:
        st.warning("❌ Nuk gjendet në dokumentet e ngarkuara.")
        st.stop()

    q_tokens = tokenize(q)
    docs_tokens = [tokenize(c["text"]) for c in candidates]
    bm25 = BM25Okapi(docs_tokens)
    bm25_scores = bm25.get_scores(q_tokens)

    for c, b in zip(candidates, bm25_scores):
        c["bm25_score"] = float(b)

    candidates.sort(key=lambda x: (x["bm25_score"], x["dense_score"]), reverse=True)
    results = candidates[:top_k]

    st.subheader("Answer")
    answer = generate_answer(q, results[:3])
    st.write(answer)

    st.subheader("Citations")
    for r in results[:3]:
        st.write(f"- {r['doc_name']} | p.{r['page']} | chunk {r['chunk_id']}")

    st.subheader("Top passages")
    for r in results:
        cite = (
            f"{r['doc_name']} | p.{r['page']} | chunk {r['chunk_id']} | "
            f"dense {r['dense_score']:.3f} | bm25 {r['bm25_score']:.2f}"
        )
        with st.expander(cite):
            st.write(r["text"])
