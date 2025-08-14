import re, math
from collections import Counter

SENTS = [
    "Kumar led backend teams using Node.js and Python on AWS and GCP.",
    "He designed CI/CD pipelines and reduced cloud costs significantly.",
    "He managed cross-functional teams and delivered features with low latency."
]

def tokenize(t): return re.findall(r"[a-z0-9]+", t.lower())

# build corpus
docs = [tokenize(s) for s in SENTS]
N = len(docs)

# document frequency (how many docs contain term)
# df[term] = number of documents that contain term
# example: df["kumar"] = 2 (because kumar is in 2 of the 3 sentences)
df = Counter()
for d in docs:
    for term in set(d):
        df[term] += 1

# idf with smoothing to avoid extremes
def idf(term):
    # +1 smoothing: log( (N + 1) / (df + 1) ) + 1  -> keeps values positive
    return math.log((N + 1) / (df[term] + 1)) + 1.0

def tfidf_vector(tokens):
    tf = Counter(tokens)          # term frequency in this doc/query
    vec = {}
    for t, c in tf.items():
        vec[t] = (c * idf(t))
    return vec

def cosine_sim(vecA, vecB):
    # dot / (||A|| * ||B||)
    common = set(vecA) & set(vecB)
    dot = sum(vecA[t] * vecB[t] for t in common)
    na = math.sqrt(sum(v*v for v in vecA.values()))
    nb = math.sqrt(sum(v*v for v in vecB.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

if __name__ == "__main__":
    # precompute TF-IDF for each sentence
    doc_vecs = [tfidf_vector(d) for d in docs]

    print("TF-IDF index ready over", N, "sentences.")
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit","quit"): break
        q_tokens = tokenize(q)
        q_vec = tfidf_vector(q_tokens)

        scores = []
        for i, v in enumerate(doc_vecs):
            s = cosine_sim(q_vec, v)
            scores.append((s, i, SENTS[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        print("\nRanking (cosine TF-IDF):")
        for s, i, sent in scores:
            print(f"{s:.3f}  ->  #{i}: {sent}")

        print("\nTop match explanation:")
        top_s, top_i, top_sent = scores[0]
        print(f"- Score: {top_s:.3f}")
        print(f"- Top sentence: {top_sent}")
