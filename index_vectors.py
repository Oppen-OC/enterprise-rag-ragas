"""Paso 2 de la ingesta: chunks.jsonl -> coleccion de Qdrant.

    python index_vectors.py data/processed/mi.chunks.jsonl --recreate

Cada punto guarda en su payload lo que el chat necesita para citar la fuente:
source_file, page_number, heading, raw_text y chunk_id.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import (
    DENSE_VECTOR,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SPARSE_VECTOR,
    embed_sparse,
    embed_texts,
)


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def embedding_text(chunk: dict) -> str:
    """Texto que se vectoriza: el chunk precedido de su encabezado.

    El encabezado da contexto a fragmentos que, sueltos, serian ambiguos
    ("provides the necessary rights..." no dice de que producto habla).
    """
    heading = chunk.get("heading", "")
    return f"{heading}\n\n{chunk['raw_text']}" if heading else chunk["raw_text"]


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            # Cada punto guarda dos vectores con nombre: el semantico y el BM25.
            vectors_config={
                DENSE_VECTOR: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            },
            # modifier=IDF hace que Qdrant calcule la parte IDF de BM25 con las
            # estadisticas de toda la coleccion, que es lo que fastembed no puede
            # saber al vectorizar un chunk aislado.
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        print(f"      coleccion '{QDRANT_COLLECTION}' creada ({EMBEDDING_DIM} dims cosine + BM25)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--recreate", action="store_true", help="borra la coleccion antes de indexar")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"[1/3] {len(chunks)} chunks leidos de {args.chunks}")

    texts = [embedding_text(c) for c in chunks]

    print("[2/3] Vectorizando: embeddings densos + BM25...")
    vectors = embed_texts(texts)
    if vectors and len(vectors[0]) != EMBEDDING_DIM:
        raise SystemExit(
            f"EMBEDDING_DIM={EMBEDDING_DIM} pero el modelo devuelve {len(vectors[0])}. "
            "Corrige EMBEDDING_DIM en el .env o la coleccion no cuadrara."
        )
    sparse = embed_sparse(texts)

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, args.recreate)

    # El id del punto se deriva del chunk_id: reindexar sobrescribe en vez de duplicar.
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector={
                DENSE_VECTOR: vector,
                SPARSE_VECTOR: SparseVector(indices=indices, values=values),
            },
            payload=chunk,
        )
        for chunk, vector, (indices, values) in zip(chunks, vectors, sparse)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    total = client.count(QDRANT_COLLECTION).count
    print(f"[3/3] {len(points)} puntos indexados. Total en la coleccion: {total}")


if __name__ == "__main__":
    main()
