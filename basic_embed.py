# rag_rank_and_answer.py
import os, math
from google import genai

# ---- 1) Data ----
docs = [
    "I love working on AWS.",
    "Latency matters in APIs.",
    "GCP and Python for data pipelines.",
    "FastAPI is a Python web framework."
]
queries = [
    "cloud platforms",
    "fast API",
    "low latency API",
    "FastAPI Python framework",
]

# ---- 2) Tiny math helpers ----
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(sum(x * x for x in a))

def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)

def normalize(v):
    n = norm(v)
    return v if n == 0.0 else [x / n for x in v]

# ---- 3) Rank helper ----
def rank_docs_for_query(query_vec, doc_vecs, k=2, assume_normalized=True):
    scores = []
    for i, dv in enumerate(doc_vecs):
        assert len(query_vec) == len(dv), "Vector length mismatch"
        s = dot(query_vec, dv) if assume_normalized else cosine(query_vec, dv)
        scores.append((s, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]

if __name__ == "__main__":
    # ---- 4) Gemini client ----
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")
    client = genai.Client(api_key=api_key)

    # ---- 5) Embed docs & queries ----
    emb_model = "text-embedding-004"
    doc_emb = client.models.embed_content(model=emb_model, contents=docs)
    query_emb = client.models.embed_content(model=emb_model, contents=queries)

    doc_vecs = [e.values for e in doc_emb.embeddings]
    query_vecs = [e.values for e in query_emb.embeddings]

    print(f"dims: {len(doc_vecs[0])} {len(query_vecs[0])}")
    print(f"docs: {len(doc_vecs)} queries: {len(query_vecs)}")

    # ---- 6) Normalize once (so dot == cosine) ----
    norm_docs = [normalize(v) for v in doc_vecs]
    norm_queries = [normalize(v) for v in query_vecs]

    # ---- 7) Rank + Answer with top-k context ----
    gen_model = "gemini-1.5-flash"
    top_k = 2

    for qi, qv in enumerate(norm_queries):
        MIN_SCORE = 0.40
        top = rank_docs_for_query(qv, norm_docs, k=top_k, assume_normalized=True)
        if not top or top[0][0] < MIN_SCORE:
            print("\nAnswer: I don't know.")
            continue

        print(f"\nQuery: {queries[qi]!r}")
        for rank, (score, idx) in enumerate(top, start=1):
            print(f"  {rank}. score={score:.3f}  →  {docs[idx]}")

        # Build one context from the top-k docs
        ctx_items = [f"[{i}] {docs[i]}" for _, i in top]
        ctx = "\n".join(f"- {s}" for s in ctx_items)

        prompt = f"""
        You are a concise assistant. Answer the question using ONLY the provided snippets.
        - If the snippets don’t contain the answer, say: "I don't know."
        - Prefer a single sentence when possible.
        - Include citations like [doc_id] for any facts you state (e.g., "[2]").

        Snippets:
        {ctx}

        Question: {queries[qi]}
        Answer (with citations):
        """.strip()

        resp = client.models.generate_content(model=gen_model, contents=prompt)
        answer = getattr(resp, "text", None) or getattr(resp, "output_text", "")
        print("Sources:", ", ".join(f"[{i}]" for _, i in top))
        print("\nAnswer:", answer.strip())