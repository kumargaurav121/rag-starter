# RAG Starter Project

This project is a learning sandbox for understanding **Retrieval-Augmented Generation (RAG)**.  
We will build a minimal pipeline that combines **embeddings**, **vector search**, and **LLMs** to create grounded answers with citations.

---

## 📌 Goal for Day 1
- Understand **embeddings** and how to measure similarity between them.
- Store embeddings in a **vector store** (e.g., FAISS).
- Run queries to retrieve the closest matching documents.
- Use retrieved chunks as **context** for an LLM to generate grounded answers.

---

## 🧩 How RAG Works

### Steps
1. **Embedding** – Convert documents and user queries into numerical vectors using Gemini (`text-embedding-004`).
2. **Indexing** – Store document embeddings in a vector store (e.g., FAISS, pgvector).
3. **Retrieval** – At query time, embed the user’s question and find the most similar documents via cosine similarity.
4. **Augmentation** – Insert retrieved documents into a prompt.
5. **Generation** – Ask an LLM (e.g., `gemini-1.5-flash`) to generate an answer based on the provided context.
6. **Output** – Return an answer with **sources** for transparency.

---

## 📊 Diagram

```mermaid
flowchart LR
  subgraph Offline[Indexing (one-time or on update)]
    D[Documents] --> ED[Embeddings\n(text-embedding-004)]
    ED --> V[Vector Store\n(FAISS/pgvector)]
  end

  subgraph Online[Query-time]
    Q[User Query] --> EQ[Embed Query\n(text-embedding-004)]
    EQ --> R[Retrieve Top-k by Cosine\nfrom Vector Store]
    R --> P[Prompt Builder\n(Question + Context)]
    P --> L[LLM\n(gemini-1.5-flash)]
    L --> A[Answer + Sources]
  end
```
---

## Author
[Kumar Gaurav]

## Diagram
```mermaid
flowchart LR
  Q[User Query] --> E[Embed]
  E --> V[Vector Index]
  V --> R[Top-k Chunks]
  R --> P[Prompt + Context]
  P --> L[LLM Answer]
