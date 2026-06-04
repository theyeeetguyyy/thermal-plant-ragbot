---
title: Thermal RAG
emoji: 🏭
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Thermal Power Plant RAG

Document-grounded assistant for thermal power plant manuals, SOPs and
performance data. Uploaded PDFs/DOCX are turned into retrieval-ready chunks
by a dynamic ingestion pipeline and answered by an OpenAI chat model.

## How complex documents are handled

OEM boiler manuals encode a lot of data as **performance curves, plotted
graphs and diagrams** (e.g. "drum / SH outlet pressure vs % BMCR"). Plain
text extraction only recovers the axis labels, not the plotted values, so
those questions used to fail.

The ingestion pipeline (`backend/ingestion.py`) now does:

1. **Per-page extraction** with PyMuPDF.
2. **Chart/diagram detection** — pages with many vector strokes or that are
   image-dominant are rendered to PNG.
3. **Vision pass** — those page images are sent to an OpenAI vision model
   which transcribes plotted curves and tables into Markdown (incl. a table
   of readings at each gridline, e.g. the value at 100% BMCR). The PNG is
   saved under `data/_images/<file>/` for reference.
4. **Markdown cleanup** (de-hyphenation, whitespace) and **structure-aware
   chunking** that respects headings.
5. **Embeddings** via OpenAI `text-embedding-3-small`, stored in Chroma.

Text/table-only pages skip the vision step, so cost stays proportional to how
visual a document is.

## Required environment variables

Set these as Space secrets (or in a local `.env`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | chat + vision + embeddings | — (required) |
| `MONGODB_URI` | users, chats, documents metadata | `mongodb://localhost:27017` |
| `MONGODB_DB` | database name | `thermal_plant_ragbot` |
| `OPENAI_CHAT_MODEL` | answer model | `gpt-4.1` |
| `OPENAI_VISION_MODEL` | chart-reading model | `gpt-4.1` |
| `OPENAI_EMBED_MODEL` | embedding model | `text-embedding-3-small` |
| `VISION_MAX_PAGES` | safety cap on vision calls per doc | `60` |

## Rebuilding the vector store

The vector store is **not** committed (embeddings are model-specific). Build
it from the seed documents in `data/` after setting `OPENAI_API_KEY`:

```bash
python ingest.py
```

Or just upload documents through the UI — uploads run the same pipeline. On
Hugging Face, mount persistent storage at `/data` so Chroma and uploads
survive restarts.
