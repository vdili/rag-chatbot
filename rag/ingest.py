import json
import re
from pathlib import Path

import fitz
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 1100
OVERLAP = 180
MIN_CHUNK_LEN = 200


def clean(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    t = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*\d+\s*$", " ", t, flags=re.MULTILINE)

    t = re.sub(r"www\.genpact\.com", " ", t, flags=re.IGNORECASE)
    t = re.sub(
        r"©\s*\d{4}\s*Copyright\s*Genpact\.?\s*All\s*Rights\s*Reserved\.?",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bAll\s*Rights\s*Reserved\.?\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*©.*$", " ", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*Copyright.*$", " ", t, flags=re.MULTILINE)

    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def chunk_text(text: str):
    out = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + CHUNK_CHARS)
        c = text[i:j].strip()
        if c:
            out.append((i, j, c))
        if j == n:
            break
        i = max(0, j - OVERLAP)

    return out


def infer_industry(filename: str) -> str:
    base = filename.lower()
    if "__" in base:
        return base.split("__", 1)[0]
    if base.startswith("insurance"):
        return "insurance"
    if base.startswith("supply"):
        return "supply"
    if base.startswith("banking"):
        return "banking"
    if base.startswith("finance"):
        return "finance"
    return "other"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit("❌ Ska PDF në data/raw.")

    model = SentenceTransformer(MODEL)

    metas = []
    texts = []
    cid = 0

    for p in tqdm(pdfs, desc="PDFs"):
        doc = fitz.open(p)
        industry = infer_industry(p.name)

        for pi in range(len(doc)):
            page = doc.load_page(pi)
            txt = page.get_text("text", sort=True)
            txt = clean(txt)

            if not txt:
                continue

            for a, b, c in chunk_text(txt):
                if len(c) < MIN_CHUNK_LEN:
                    continue

                metas.append(
                    {
                        "chunk_id": cid,
                        "doc_name": p.name,
                        "industry": industry,
                        "page": pi + 1,
                        "char_start": a,
                        "char_end": b,
                        "text": c,
                    }
                )
                texts.append(c)
                cid += 1

        doc.close()

    if not metas:
        raise SystemExit("❌ Nuk u nxor tekst nga PDF-të.")

    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.array(embs, dtype=np.float32)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with (INDEX_DIR / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf_count": len(pdfs),
                "chunk_count": len(metas),
                "model": MODEL,
                "chunk_chars": CHUNK_CHARS,
                "overlap_chars": OVERLAP,
                "min_chunk_len": MIN_CHUNK_LEN,
            },
            f,
            indent=2,
        )

    print(f"✅ DONE: {len(pdfs)} PDFs | {len(metas)} chunks")


if __name__ == "__main__":
    main()
