"""Tests de retrieval.py contra un Qdrant en memoria: sin Docker, sin Ollama."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

import retrieval


def _chunk(name: str) -> dict:
    return {"source_file": f"{name}.pdf", "page_number": 1, "heading": name, "raw_text": f"texto {name}"}


@pytest.fixture
def qdrant(monkeypatch):
    """Tres chunks cuyo orden denso y cuyo orden BM25 NO coinciden.

        denso (coseno con la consulta [1, 0]):  A > B > C
        BM25  (termino 1 de la consulta):       B > C   (A no lo contiene)

    B es el unico bien situado en las dos ramas, asi que RRF lo sube al
    primer puesto aunque en denso sea segundo.
    """
    client = QdrantClient(":memory:")
    client.create_collection(
        retrieval.QDRANT_COLLECTION,
        vectors_config={retrieval.DENSE_VECTOR: VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={retrieval.SPARSE_VECTOR: SparseVectorParams()},
    )
    points = [
        ("A", [1.0, 0.0], SparseVector(indices=[2], values=[1.0])),
        ("B", [0.8, 0.6], SparseVector(indices=[1], values=[1.0])),
        ("C", [0.0, 1.0], SparseVector(indices=[1], values=[0.1])),
    ]
    client.upsert(
        retrieval.QDRANT_COLLECTION,
        [
            PointStruct(id=i, vector={retrieval.DENSE_VECTOR: dense, retrieval.SPARSE_VECTOR: sparse}, payload=_chunk(name))
            for i, (name, dense, sparse) in enumerate(points, start=1)
        ],
    )
    monkeypatch.setattr(retrieval, "_client", client)
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[1.0, 0.0]])
    monkeypatch.setattr(retrieval, "embed_sparse_query", lambda q: ([1], [1.0]))


def _headings(mode: str, monkeypatch, k: int = 3) -> list[str]:
    monkeypatch.setattr(retrieval, "RETRIEVAL_MODE", mode)
    return [c["heading"] for c in retrieval.retrieve("pregunta", k=k)]


def test_dense_ordena_solo_por_similitud_semantica(qdrant, monkeypatch):
    assert _headings("dense", monkeypatch) == ["A", "B", "C"]


def test_hybrid_fusiona_las_dos_ramas_con_rrf(qdrant, monkeypatch):
    assert _headings("hybrid", monkeypatch)[0] == "B"


def test_retrieve_devuelve_payload_y_score_y_respeta_k(qdrant, monkeypatch):
    monkeypatch.setattr(retrieval, "RETRIEVAL_MODE", "hybrid")
    chunks = retrieval.retrieve("pregunta", k=2)
    assert len(chunks) == 2
    assert {"source_file", "page_number", "raw_text", "score"} <= chunks[0].keys()


CHUNKS = [
    {"source_file": "a.pdf", "page_number": 49, "raw_text": "primero"},
    {"source_file": "b.pdf", "page_number": 50, "raw_text": "segundo"},
]


def test_format_context_numera_los_chunks_desde_1():
    ctx = retrieval.format_context(CHUNKS)
    assert ctx.startswith("[1] (a.pdf, pagina 49)\nprimero")
    assert "\n\n[2] (b.pdf, pagina 50)\nsegundo" in ctx


def test_build_messages_pone_contexto_y_pregunta_en_el_mensaje_de_usuario():
    system, user = retrieval.build_messages("¿que version?", CHUNKS)
    assert system == {"role": "system", "content": retrieval.SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert user["content"].endswith("Pregunta: ¿que version?")
    assert retrieval.format_context(CHUNKS) in user["content"]
