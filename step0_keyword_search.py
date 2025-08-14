import re

# 1) Our "knowledge base" = 3 simple sentences
SENTS = [
    "Kumar led backend teams using Node.js and Python on AWS and GCP.",
    "He designed CI/CD pipelines and reduced cloud costs significantly.",
    "He managed cross-functional teams and delivered features with low latency."
]

# 2) Tiny tokenizer: lower-case + keep only letters/numbers
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

# 3) Naive keyword score = how many query tokens appear in the sentence tokens
def score(query_tokens, sent_tokens):
    q = set(query_tokens)
    s = set(sent_tokens)
    overlap = q.intersection(s)
    return len(overlap), sorted(list(overlap))

if __name__ == "__main__":
    # Pre-tokenize sentences once
    tokenized = [tokenize(s) for s in SENTS]
    print(tokenized)
    print("Loaded", len(SENTS), "sentences")
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"): break

        q_tokens = tokenize(q)
        if not q_tokens:
            print("Please type a few words.")
            continue

        # Score each sentence
        scored = []
        for i, s_tokens in enumerate(tokenized):
            count, overlap = score(q_tokens, s_tokens)
            scored.append((count, overlap, i, SENTS[i]))

        # Sort by keyword-overlap (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]

        # Show ranking
        print("\nRanking (overlap_count, overlap_words, sentence_index):")
        for count, overlap, idx, sent in scored:
            print(f"{count:>2}  {overlap}  ->  #{idx}: {sent}")

        # Explain the top match
        print("\nTop match explanation:")
        print(f"- Query tokens: {q_tokens}")
        print(f"- Overlap with top sentence: {best[1]}")
