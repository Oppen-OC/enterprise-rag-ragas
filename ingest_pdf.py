"""Paso 1 de la ingesta: PDF -> markdown -> chunks.

    python ingest_pdf.py data/raw/mi.pdf --page-offset 43

Produce dos ficheros en data/processed/:

    <nombre>.md            markdown completo, con marcas de pagina. Sirve para
                           revisar a ojo si las tablas se han extraido bien.
    <nombre>.chunks.jsonl  un chunk por linea, listo para index_vectors.py.

El paso caro (extraer el PDF) se cachea en el .md: al reejecutar sin --force se
reutiliza. Eso hace el script idempotente y barato de reintentar, que es lo que
necesitaria un DAG de Airflow/Prefect si mas adelante orquestamos la ingesta.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import PROCESSED_DIR

# Marca que separa paginas dentro del .md cacheado.
PAGE_MARK = re.compile(r"^<!-- page: (\d+) -->$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Extraccion: PDF -> [(numero_de_pagina, markdown)]
# ---------------------------------------------------------------------------
def extract_with_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Backend ligero. Rapido, sin modelos, tablas en markdown decentes."""
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    return [(p["metadata"]["page_number"], p["text"]) for p in pages]


def extract_with_marker(pdf_path: Path) -> list[tuple[int, str]]:
    """Backend del enunciado. Mejor calidad en tablas, pero descarga modelos
    (~2 GB) y en CPU tarda del orden de segundos por pagina."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"paginate_output": True, "output_format": "markdown"},
    )
    text, _, _ = text_from_rendered(converter(str(pdf_path)))

    # Marker separa paginas con una linea "{n}------...". La convertimos al
    # mismo formato (numero, markdown) que devuelve el otro backend.
    parts = re.split(r"^\{(\d+)\}-{10,}$", text, flags=re.MULTILINE)
    pages: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        pages.append((int(parts[i]) + 1, parts[i + 1].strip()))
    return pages or [(1, text)]


BACKENDS = {"pymupdf": extract_with_pymupdf, "marker": extract_with_marker}


# ---------------------------------------------------------------------------
# Cache del markdown
# ---------------------------------------------------------------------------
def write_markdown(pages: list[tuple[int, str]], path: Path) -> None:
    blocks = [f"<!-- page: {number} -->\n\n{text}\n" for number, text in pages]
    path.write_text("\n".join(blocks), encoding="utf-8")


def read_markdown(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(encoding="utf-8")
    parts = PAGE_MARK.split(raw)
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def build_chunker(kind: str, chunk_size: int):
    """El chunker recursivo corta por la estructura del markdown (encabezados,
    parrafos, filas), asi que respeta tablas; el de tokens da el solape del 10%
    que pide el enunciado a costa de poder partir una tabla por la mitad."""
    from chonkie import RecursiveChunker, TokenChunker

    if kind == "recursive":
        return RecursiveChunker.from_recipe("markdown", lang="en", tokenizer="gpt2", chunk_size=chunk_size)
    return TokenChunker(tokenizer="gpt2", chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1))


SEPARATOR_ROW = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def compact_tables(markdown: str) -> str:
    """Quita el relleno de espacios con el que Marker alinea las columnas.

    Una tabla alineada puede ocupar el triple de tokens que la misma tabla sin
    relleno, lo bastante para pasarse del chunk_size y acabar partida. Al
    chunker le damos la version compacta; el .md conserva la bonita.
    """
    lines = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("|"):
            cells = [re.sub(r"\s+", " ", cell).strip() for cell in line.split("|")]
            cells = ["---" if re.fullmatch(r":?-{2,}:?", c) else c for c in cells]
            line = "|".join(cells)
        lines.append(line)
    return "\n".join(lines)


def last_table_header(markdown: str) -> str | None:
    """Cabecera (fila de titulos + separador) de la ultima tabla que empieza en el texto."""
    lines = markdown.splitlines()
    header = None
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and SEPARATOR_ROW.match(lines[i + 1].strip()):
            header = lines[i] + "\n" + lines[i + 1]
    return header


def repair_table_header(text: str, markdown_before: str) -> str:
    """Si un chunk arranca en mitad de una tabla, le devuelve su cabecera.

    Sin esto, la segunda mitad de una tabla partida son filas sueltas sin nombre
    de columna, y el LLM confunde unas columnas con otras.
    """
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("|"):
        return text
    if len(lines) > 1 and SEPARATOR_ROW.match(lines[1].strip()):
        return text  # ya trae su cabecera
    header = last_table_header(markdown_before)
    return header + "\n" + text if header else text


def current_heading(markdown: str, previous: str) -> str:
    """Ultimo encabezado markdown de un texto, o el que veniamos arrastrando.

    Se guarda junto al chunk para que un fragmento suelto en mitad de una
    seccion siga sabiendo de que seccion habla al recuperarlo.
    """
    headings = re.findall(r"^#{1,3} +(.+)$", markdown, flags=re.MULTILINE)
    return headings[-1].strip() if headings else previous


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    chunker,
    page_offset: int,
) -> list[dict]:
    records: list[dict] = []
    heading = ""
    for page_number, markdown in pages:
        if not markdown.strip():
            continue
        heading = current_heading(markdown, heading)
        markdown = compact_tables(markdown)
        for i, chunk in enumerate(chunker(markdown)):
            text = repair_table_header(chunk.text.strip(), markdown[: chunk.start_index])
            if not text:
                continue
            records.append(
                {
                    "chunk_id": f"{Path(source_file).stem}-p{page_number}-c{i}",
                    "source_file": source_file,
                    "page_number": page_number + page_offset,
                    "heading": heading,
                    "raw_text": text,
                }
            )
    return records


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, default="marker")
    parser.add_argument("--chunker", choices=("recursive", "token"), default="recursive")
    parser.add_argument("--chunk-size", type=int, default=500, help="tokens por chunk")
    parser.add_argument(
        "--page-offset",
        type=int,
        default=0,
        help="se suma al numero de pagina; util si el PDF es un recorte de otro mayor",
    )
    parser.add_argument("--force", action="store_true", help="reextrae aunque exista el .md")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    md_path = PROCESSED_DIR / f"{args.pdf.stem}.md"
    jsonl_path = PROCESSED_DIR / f"{args.pdf.stem}.chunks.jsonl"

    if md_path.exists() and not args.force:
        print(f"[1/2] Reutilizando {md_path} (usa --force para reextraer)")
        pages = read_markdown(md_path)
    else:
        print(f"[1/2] Extrayendo {args.pdf} con backend '{args.backend}'...")
        pages = BACKENDS[args.backend](args.pdf)
        write_markdown(pages, md_path)
        print(f"      {len(pages)} paginas -> {md_path}")

    print(f"[2/2] Troceando con chunker '{args.chunker}' ({args.chunk_size} tokens)...")
    records = chunk_pages(pages, args.pdf.name, build_chunker(args.chunker, args.chunk_size), args.page_offset)

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"      {len(records)} chunks -> {jsonl_path}")


if __name__ == "__main__":
    main()
