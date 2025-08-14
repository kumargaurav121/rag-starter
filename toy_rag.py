import os, sys, pdfplumber, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PDF = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
CHUNK=600; OVERLAP=120; TOPK=3
MODEL="sentence-transformers/all-MiniLM-L6-v2"

def load_pdf(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    txt=[]
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages: txt.append(p.extract_text() or "")
    return "\n".join(txt)

def chunks(s, size, overlap):
    out=[]; i=0
    while i < len(s):
        c=s[i:i+size].strip()
        if c: out.append(c)
        i += max(1, size-overlap)
    return out

def build(chunks_, model):
    emb = model.encode(chunks_, convert_to_numpy=True, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized vectors
    idx.add(emb)
    return idx, emb

def search(q, model, idx, chunks_, k):
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = idx.search(qv, k)
    return [(float(s), int(i), chunks_[int(i)]) for s,i in zip(scores[0], ids[0])]

if __name__=="__main__":
    text = load_pdf(PDF)
    parts = chunks(text, CHUNK, OVERLAP)
    model = SentenceTransformer(MODEL)
    idx, _ = build(parts, model)
    print(f"Loaded {len(parts)} chunks. Ask questions (type 'exit' to quit).")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit","quit"): break
        for r,(score,i,chunk) in enumerate(search(q, model, idx, parts, TOPK),1):
            print(f"[{r}] score={score:.3f} chunk#{i}\n{chunk[:400]}\n---")
