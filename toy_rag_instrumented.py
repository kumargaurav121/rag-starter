
# toy_rag_instrumented.py
# Verbose "explain mode" for today's RAG toy.
# Run: python toy_rag_instrumented.py resume.pdf
# If you don't have a PDF handy, rename any PDF to resume.pdf or pass a path.

import os, sys, re, statistics
import pdfplumber, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PDF = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
CHUNK=600; OVERLAP=120; TOPK=3
MODEL="sentence-transformers/all-MiniLM-L6-v2"

def load_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    txt=[]
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            txt.append(t)
    return "\n".join(txt)

def chunks(s, size, overlap):
    out=[]; i=0
    while i < len(s):
        c = s[i:i+size].strip()
        if c: out.append(c)
        i += max(1, size-overlap)
    return out

def highlight_overlap(text, query):
    # naive highlighter: wrap query words found in text with [WORD]
    q_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2}
    def repl(m):
        w = m.group(0)
        return f"[{w}]" if w.lower() in q_words else w
    return re.sub(r"[A-Za-z0-9]+", repl, text)

if __name__ == "__main__":
    print("=== STEP 1: Load text from PDF ===")
    text = load_pdf(PDF)
    print(f"Total characters: {len(text)}")
    print("\nFirst 400 chars:\n", text[:400].replace("\n"," "), "\n")

    print("=== STEP 2: Chunking ===")
    parts = chunks(text, CHUNK, OVERLAP)
    lens = [len(p) for p in parts]
    print(f"Chunks created: {len(parts)}  (size={CHUNK}, overlap={OVERLAP})")
    if parts:
        import statistics as _stats
        print(f"Chunk length stats: min={min(lens)}, median={int(_stats.median(lens))}, max={max(lens)}")
        print("\nPreview chunk#0:\n", parts[0][:300].replace("\n"," "), "\n")
        if len(parts) > 1:
            print("Preview chunk#1:\n", parts[1][:300].replace("\n"," "), "\n")

    print("=== STEP 3: Embeddings & Index ===")
    model = SentenceTransformer(MODEL)
    emb = model.encode(parts, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1] if len(emb) else 0
    print(f"Embedding model: {MODEL}  |  vector dim = {dim}")
    if len(emb):
        print("First 8 dims of first vector:", np.round(emb[0][:8], 4).tolist())
    index = faiss.IndexFlatIP(dim)  # cosine similarity because vectors normalized (inner product == cosine)
    if len(emb):
        index.add(emb)
    print(f"Index size: {index.ntotal}")

    print("\n=== STEP 4: Query time ===")
    print("Type a question about your resume (or 'exit')")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit","quit"): break
        if not len(emb):
            print("No chunks to search. Check your PDF text extraction.")
            continue
        qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = index.search(qv, TOPK)
        print("\nTop results (cosine similarity; closer to 1.0 = more similar):")
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
            chunk = parts[int(idx)]
            print(f"[{rank}] score={float(score):.3f}  chunk#{int(idx)}")
            # simple query word highlighting
            import re as _re
            q_words = {w.lower() for w in _re.findall(r"[A-Za-z0-9]+", q) if len(w) > 2}
            def _repl(m):
                w = m.group(0)
                return f"[{w}]" if w.lower() in q_words else w
            print(_re.sub(r"[A-Za-z0-9]+", _repl, chunk[:400]))
            print("---")
        print()
