"""Recuperacion: unico punto donde se decide QUE contexto ve el LLM.

Dos modos, elegidos con RETRIEVAL_MODE en el .env:

  dense   busqueda semantica pura.
  hybrid  semantica + BM25, fusionadas con Reciprocal Rank Fusion.

app.py y evaluate_rag.py no distinguen entre ambos: solo conocen la firma
retrieve(pregunta, k) -> lista de chunks.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from config import (
    DENSE_VECTOR,
    PREFETCH_LIMIT,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RETRIEVAL_MODE,
    SPARSE_VECTOR,
    TOP_K,
    embed_sparse_query,
    embed_texts,
)

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _dense_search(question: str, k: int):
    return get_qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        query=embed_texts([question])[0],
        using=DENSE_VECTOR,
        limit=k,
        with_payload=True,
    ).points


def _hybrid_search(question: str, k: int):
    """Cada rama busca por su cuenta y Qdrant fusiona las dos listas con RRF.

    RRF puntua por POSICION, no por score: un chunk que sale 1o en BM25 y 8o en
    denso acaba arriba aunque sus dos scores no sean comparables entre si. Eso
    es justo lo que hace falta, porque una distancia coseno y un peso BM25 estan
    en escalas distintas y no se pueden sumar.
    """
    indices, values = embed_sparse_query(question)
    return get_qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=embed_texts([question])[0],
                using=DENSE_VECTOR,
                limit=PREFETCH_LIMIT,
            ),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR,
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    ).points


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Devuelve los k chunks mas relevantes para la pregunta, mejor primero."""
    search = _hybrid_search if RETRIEVAL_MODE == "hybrid" else _dense_search
    return [{**hit.payload, "score": hit.score} for hit in search(question, k)]


def format_context(chunks: list[dict]) -> str:
    """Numera los chunks para que el LLM pueda citarlos como [1], [2]..."""
    return "\n\n".join(
        f"[{i}] ({chunk['source_file']}, pagina {chunk['page_number']})\n{chunk['raw_text']}"
        for i, chunk in enumerate(chunks, start=1)
    )


SYSTEM_PROMPT = """Eres un asistente que responde SOLO con la informacion de los fragmentos suministrados.

Reglas:
- Si la respuesta no esta en los fragmentos, di que no aparece en la documentacion. No inventes.
- Cita los fragmentos que uses con su numero entre corchetes, por ejemplo [1].
- Si el dato viene de una tabla, conserva la estructura de la tabla en la respuesta.
- Responde en el idioma de la pregunta."""


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fragmentos:\n\n{format_context(chunks)}\n\nPregunta: {question}"},
    ]
