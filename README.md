# Module 5: Advanced AI Architectures

This module covers machine learning systems at scale, including MLOps pipelines (training, deployment, monitoring), hosting Large Language Models (LLMs), retrieval-augmented generation (RAG) systems, and agentic workflows.

## Exercises

*   [**Exercise 1: Building Enterprise Retrieval-Augmented Generation (RAG)**](./exercise-1-building-enterprise-rag.md)
    *   Construct a production-grade RAG pipeline using Marker, Chonkie, Qdrant, and Chainlit, with optional steps for LiteLLM, hybrid search, and Langfuse observability.
*   [**Exercise 2: Evaluating RAG Pipelines with Ragas**](./exercise-2-ragas-evaluation.md)
    *   Perform a quickstart evaluation using the Ragas framework to understand key metrics like Faithfulness, Answer Relevance, Context Recall, and Context Precision.

## Solution

Both exercises are solved in this directory. Setup, run instructions, architecture
diagram and design decisions: **[README-implementacion.md](./README-implementacion.md)**.

| File | Responsibility |
| --- | --- |
| [`config.py`](./config.py) | Single place that knows the LLM/embedding provider and paths |
| [`ingest_pdf.py`](./ingest_pdf.py) | PDF → markdown (cached) → `chunks.jsonl` |
| [`index_vectors.py`](./index_vectors.py) | `chunks.jsonl` → Qdrant `enterprise_docs` collection |
| [`retrieval.py`](./retrieval.py) | Dense or hybrid (BM25 + RRF) retrieval and prompt assembly |
| [`app.py`](./app.py) | Chainlit chat with streaming and citations |
| [`evaluate_rag.py`](./evaluate_rag.py) | Exercise 2: Ragas metrics |

---
[⬅ Back to Home](../README.md)
