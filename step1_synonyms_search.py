import re
SENTS = [
    "Kumar led backend teams using Node.js and Python on AWS and GCP.",
    "He designed CI/CD pipelines and reduced cloud costs significantly.",
    "He managed cross-functional teams and delivered features with low latency."
]

def tokenize(text):  # lowercase, alnum only, split on non-alnum
    return re.findall(r"[a-z0-9]+", text.lower())

# mini-thesaurus for demo (expand as you wish)
SYN = {
    "cloud": ["cloud", "aws", "gcp", "azure"],
    "cost": ["cost", "costs", "spend", "spending", "optimize", "optimization"],
    "team": ["team", "teams", "cross", "functional", "crossfunctional"],
    "latency": ["latency", "p95", "performance"]
}

def expand_query_tokens(q_tokens):
    expanded = set()
    for t in q_tokens:
        expanded.update(SYN.get(t, [t]))  # if we have synonyms, use them; else keep token
    return expanded

if __name__ == "__main__":
    tokenized = [tokenize(s) for s in SENTS]
    print("Loaded", len(SENTS), "sentences.")
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit","quit"): break
        q_tokens = tokenize(q)
        if not q_tokens: 
            print("Type a few words."); 
            continue

        expanded = expand_query_tokens(q_tokens)
        scored = []
        for i, s_tokens in enumerate(tokenized):
            overlap = set(s_tokens).intersection(expanded)
            scored.append((len(overlap), sorted(list(overlap)), i, SENTS[i]))

        scored.sort(key=lambda x: x[0], reverse=True)
        print("\nRanking (overlap_count, overlap_words, idx):")
        for count, overlap, idx, sent in scored:
            print(f"{count:>2}  {overlap}  ->  #{idx}: {sent}")

        print("\nTop match explanation:")
        top = scored[0]
        print(f"- Raw query tokens: {q_tokens}")
        print(f"- Expanded tokens:  {sorted(list(expanded))}")
        print(f"- Overlap with top: {top[1]}")
