"""Dynamic document ingestion pipeline.

Turns an uploaded PDF/DOCX into clean, retrieval-ready chunks:

    PDF  ->  per-page extraction
             |- text/table pages: native text -> markdown
             |- chart/image pages: rendered to PNG, read by an OpenAI
                vision model that transcribes plotted values + tables
                into markdown, and the PNG is saved for later viewing
         ->  assemble + clean markdown
         ->  structure-aware chunking (page + type metadata)
         ->  LangChain Documents (embedded by the caller)

The vision step is what lets the bot answer questions whose answer only
exists inside a graph (e.g. "drum pressure at 100% BMCR"), which plain
text extraction can never recover.
"""

from __future__ import annotations

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- tunables (override via env) -------------------------------------------
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1")
# A page is treated as a chart/diagram when it has at least this many vector
# draw ops (plotted curves, gridlines, schematics).
DRAWING_THRESHOLD = int(os.environ.get("VISION_DRAWING_THRESHOLD", "40"))
# Image-dominant page with little text (scanned figure) also goes to vision.
IMAGE_TEXT_FLOOR = int(os.environ.get("VISION_IMAGE_TEXT_FLOOR", "200"))
# Safety cap on vision calls per document. Pages beyond this are kept as
# native text and the shortfall is logged (never silently dropped).
MAX_VISION_PAGES = int(os.environ.get("VISION_MAX_PAGES", "300"))
# How many vision (network) calls to run concurrently.
VISION_CONCURRENCY = int(os.environ.get("VISION_CONCURRENCY", "6"))
RENDER_SCALE = float(os.environ.get("VISION_RENDER_SCALE", "2.0"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

VISION_PROMPT = (
    "You are transcribing a single page from a thermal power plant engineering "
    "manual into Markdown for a search index. Reproduce ALL information on the "
    "page faithfully and concisely.\n\n"
    "Rules:\n"
    "- If the page is a CHART or GRAPH: state the title, the X axis (name + "
    "units + range) and Y axis (name + units + range). Then read the plotted "
    "curve(s) and output a Markdown table of values at each labelled gridline "
    "of the X axis, INCLUDING the endpoints (e.g. at 100% load/BMCR). If there "
    "are multiple curves, give one column per curve. Note these are approximate "
    "readings from the graph.\n"
    "- If the page is a TABLE or data sheet: reproduce it as a Markdown table, "
    "keeping every row/column label and value.\n"
    "- If the page is a DIAGRAM/schematic: list the labelled components and any "
    "annotated values.\n"
    "- Preserve any specifications, parameters and units exactly.\n"
    "- Do not invent values you cannot see. Output only the Markdown, no preamble."
)


def _is_visual_page(page: "fitz.Page") -> bool:
    """Chart / diagram / image-dominant pages need a vision pass."""
    try:
        drawings = len(page.get_drawings())
    except Exception:
        drawings = 0
    images = len(page.get_images(full=True))
    text_len = len((page.get_text() or "").strip())
    if drawings >= DRAWING_THRESHOLD:
        return True
    if images >= 1 and text_len < IMAGE_TEXT_FLOOR:
        return True
    return False


def _render_png(page: "fitz.Page") -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
    return pix.tobytes("png")


def _vision_markdown(client: Any, png_bytes: bytes) -> str:
    """Send a page image to the OpenAI vision model and get Markdown back."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _clean_markdown(md: str) -> str:
    """Light cleanup: de-hyphenate line breaks, collapse blank runs/space."""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # join words split across line breaks: "perfor-\nmance" -> "performance"
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)
    # collapse 3+ newlines to a paragraph break
    md = re.sub(r"\n{3,}", "\n\n", md)
    # trim trailing spaces on each line
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md.strip()


def process_pdf(
    file_path: str,
    source_file: str,
    images_dir: str | None = None,
    client: Any = None,
) -> tuple[list[Document], dict[str, int]]:
    """Process a PDF into LangChain Documents (one per page) + stats.

    Pages are classified as text vs chart/diagram. Chart pages are rendered to
    PNG in the main thread (PyMuPDF is not thread-safe across one Document) and
    their vision calls — the slow, network-bound part — run concurrently.
    """
    doc = fitz.open(file_path)
    n = doc.page_count
    stats = {
        "pages": n,
        "vision_used": 0,
        "vision_skipped_cap": 0,
        "vision_failed": 0,
    }

    native_text: list[str] = []
    visual_idxs: list[int] = []
    for i in range(n):
        page = doc[i]
        native_text.append((page.get_text() or "").strip())
        if _is_visual_page(page):
            visual_idxs.append(i)

    # Apply the cap explicitly and loudly — capped pages keep native text.
    do_vision = visual_idxs
    if client is None:
        do_vision = []
    elif len(visual_idxs) > MAX_VISION_PAGES:
        stats["vision_skipped_cap"] = len(visual_idxs) - MAX_VISION_PAGES
        print(
            f"[ingestion] {source_file}: {len(visual_idxs)} chart/visual pages > "
            f"cap {MAX_VISION_PAGES}; {stats['vision_skipped_cap']} pages kept as "
            f"native text. Raise VISION_MAX_PAGES to cover them."
        )
        do_vision = visual_idxs[:MAX_VISION_PAGES]

    # Render (main thread) + persist each chart page once.
    if do_vision and images_dir:
        os.makedirs(images_dir, exist_ok=True)
    rendered: dict[int, bytes] = {}
    for i in do_vision:
        png = _render_png(doc[i])
        rendered[i] = png
        if images_dir:
            with open(os.path.join(images_dir, f"page_{i + 1}.png"), "wb") as fh:
                fh.write(png)

    # Vision calls (network-bound) run concurrently.
    vision_md: dict[int, str] = {}

    def _call(i: int) -> tuple[int, str | None]:
        try:
            return i, _vision_markdown(client, rendered[i])
        except Exception as exc:  # noqa: BLE001 — fall back to native text
            print(f"[ingestion] {source_file} page {i + 1} vision failed: {exc}")
            return i, None

    if do_vision:
        workers = max(1, min(VISION_CONCURRENCY, len(do_vision)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, md in ex.map(_call, do_vision):
                if md:
                    vision_md[i] = md
                    stats["vision_used"] += 1
                else:
                    stats["vision_failed"] += 1

    documents: list[Document] = []
    for i in range(n):
        if i in vision_md:
            md, kind = vision_md[i], "chart"
        else:
            md, kind = native_text[i], "text"
        md = _clean_markdown(md)
        if not md or len(md.strip()) < 3:
            continue
        documents.append(
            Document(
                page_content=f"## Page {i + 1}\n\n{md}",
                metadata={
                    "source_file": source_file,
                    "source": file_path,
                    "page": i,
                    "kind": kind,
                },
            )
        )
    doc.close()
    return documents, stats


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Structure-aware splitting that respects markdown boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_file(
    file_path: str,
    source_file: str,
    images_root: str | None = None,
    client: Any = None,
) -> tuple[list[Document], dict[str, int]]:
    """Full pipeline for one file -> chunked, embed-ready Documents + stats.

    DOCX files use plain text extraction (no charts to read); PDFs use the
    hybrid text + vision pipeline above.
    """
    lower = source_file.lower()
    if lower.endswith(".docx"):
        from langchain_community.document_loaders import Docx2txtLoader

        pages = Docx2txtLoader(file_path).load()
        for d in pages:
            d.metadata["source_file"] = source_file
            d.metadata["kind"] = "text"
        pages = [d for d in pages if (d.page_content or "").strip()]
        stats = {"pages": len(pages), "vision_used": 0, "vision_skipped_cap": 0, "vision_failed": 0}
        return chunk_documents(pages), stats

    images_dir = (
        os.path.join(images_root, re.sub(r"[^A-Za-z0-9._-]", "_", source_file))
        if images_root
        else None
    )
    pages, stats = process_pdf(file_path, source_file, images_dir, client)
    return chunk_documents(pages), stats
