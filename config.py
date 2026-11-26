"""Punto unico de configuracion y de acceso al proveedor de LLM/embeddings.

Ningun otro fichero crea clientes ni conoce URLs de proveedor. Eso es lo que
permite que las tareas opcionales entren sin refactor:

  - Meter LiteLLM por delante  -> cambiar OPENAI_BASE_URL en el .env.
  - Meter Langfuse             -> envolver el cliente en get_client().
  - Cambiar de modelo          -> cambiar LLM_MODEL / EMBEDDING_MODEL en el .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

# --- Rutas del modulo ------------------------------------------------------
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# --- Proveedor de LLM y embeddings ----------------------------------------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# --- Vector store ----------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_docs")
TOP_K = int(os.getenv("TOP_K", "4"))

# Nombres de los dos vectores que guarda cada punto.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# 'dense'  = solo busqueda semantica.
# 'hybrid' = semantica + BM25 fusionadas con RRF.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# Cuantos candidatos aporta cada rama antes de fusionar. Mas alto da mejor
# recall a costa de latencia; 20 es suficiente para un corpus de este tamano.
PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "20"))

# Modelo BM25 de fastembed. Corre en local sobre ONNX, no es una red neuronal:
# solo calcula frecuencias de termino, asi que es practicamente instantaneo.
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")


def get_client() -> OpenAI:
    """Cliente compatible con la API de OpenAI (Ollama, OpenAI o LiteLLM)."""
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def get_async_client() -> AsyncOpenAI:
    """Version asincrona del mismo cliente, para el streaming de Chainlit."""
    return AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Vectoriza una lista de textos con el modelo de embeddings configurado.

    La consulta del usuario y los chunks del corpus pasan por esta misma
    funcion: si los vectores no se generasen con el mismo modelo, las
    distancias en Qdrant no significarian nada.
    """
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # La API no garantiza el orden de la respuesta, asi que ordenamos por indice.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


_sparse_model = None


def get_sparse_model():
    """Modelo BM25, cargado una sola vez por proceso."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def embed_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Vectores dispersos BM25 de los chunks, como pares (indices, valores).

    A diferencia del embedding denso, aqui cada dimension es un termino
    concreto: por eso recupera coincidencias literales que el denso diluye.
    """
    return [(e.indices.tolist(), e.values.tolist()) for e in get_sparse_model().embed(texts)]


def embed_sparse_query(query: str) -> tuple[list[int], list[float]]:
    """Vector disperso de una consulta.

    fastembed usa un metodo distinto para consultas que para documentos: BM25
    no pondera igual los terminos de la pregunta que los del corpus.
    """
    embedding = next(iter(get_sparse_model().query_embed(query)))
    return embedding.indices.tolist(), embedding.values.tolist()
