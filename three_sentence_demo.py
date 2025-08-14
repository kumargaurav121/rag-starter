
# three_sentence_demo.py
# Minimal retrieval on 3 short sentences (no PDFs, no chunking).
# Run: python three_sentence_demo.py

from sentence_transformers import SentenceTransformer
import numpy as np

SENTS = [
    "Kumar led backend teams using Node.js and Python on AWS and GCP.",
    "He designed CI/CD pipelines and reduced cloud costs significantly.",
    "He managed cross-functional teams and delivered features with low latency."
]

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

if __name__ == "__main__":
    model = SentenceTransformer(MODEL)
    emb = model.encode(SENTS, convert_to_numpy=True)
    print(f"Embedded {len(SENTS)} sentences. Vector dim = {emb.shape[1]}")
    while True:
        q = input("Ask a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"): break
        qv = model.encode([q], convert_to_numpy=True)[0]
        sims = [cosine_sim(qv, e) for e in emb]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        print("\nTop match:")
        idx, score = ranked[0]
        print(f"score={score:.3f}  sentence={SENTS[idx]}")
        print("Ranking:", [(i, round(s,3)) for i,s in ranked], "\n")
