
# Day 1 — Guided Worksheet (Understanding RAG Retrieval)

## A. Run order
1) Run the tiny demo first: `python three_sentence_demo.py`
   - Ask these 3 queries and record the top match and score:
     a) "cloud platforms used?"
     b) "cost optimization"
     c) "cross-functional teams"
2) Then run the verbose PDF script: `python toy_rag_instrumented.py resume.pdf`
   - Ask any 3 queries about your own resume and paste the top result snippet for each.

## B. Short questions (answer in 1 line each)
1) What does an embedding vector represent?
2) Why does normalizing vectors before inner product approximate cosine similarity?
3) What did changing CHUNK (e.g., 400 vs 800) do to your recall?
4) What does OVERLAP help with?
5) What does a score closer to 1.0 indicate? What about near 0 or negative?

## C. Debug checklist (tick the first one that fixed things)
- [ ] PDF had no extractable text. I exported a text-based PDF or used a different extractor.
- [ ] Chunks were too big; I reduced CHUNK from 800 → 600 (or 400).
- [ ] Overlap was too small; I increased OVERLAP from 60 → 120.
- [ ] Queries were vague; I tried more specific terms that appear in my resume.
- [ ] I verified the index dimension matches the model (no errors).
- [ ] Other (write): _________

## D. Proof-of-understanding
Write **two sentences** in your own words:
- Sentence 1: What is RAG retrieval doing under the hood?
- Sentence 2: How do you know if retrieval worked for your question?
