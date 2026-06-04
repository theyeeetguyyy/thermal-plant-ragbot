"""Batch (re)builder for the Chroma vector store.

Runs the same dynamic pipeline used at upload time (PDF -> markdown, with a
vision pass over charts/diagrams) over every PDF/DOCX under ``data/`` and
writes the embeddings to ``chroma_db/``.

Requires OPENAI_API_KEY in the environment (or a .env file). Seed documents
are tagged with the DEFAULT administration code so they are visible to the
default workspace.

    python ingest.py
"""

import glob
import os
import shutil

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from backend import ingestion

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DATASET_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
IMAGES_ROOT = os.path.join(DATASET_DIR, "_images")
DEFAULT_ADMIN_CODE = "DEFAULT"
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Export it or add it to .env first.")

    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pdf"), recursive=True)
    files += glob.glob(os.path.join(DATASET_DIR, "**", "*.docx"), recursive=True)
    files = [f for f in files if os.sep + "_images" + os.sep not in f]
    if not files:
        print(f"No PDF/DOCX files found under {DATASET_DIR}/.")
        return

    print(f"Found {len(files)} file(s). Beginning processing...")
    client = OpenAI(api_key=api_key)

    all_chunks = []
    for path in files:
        source_file = os.path.basename(path)
        print(f"Processing {source_file} ...")
        try:
            chunks, stats = ingestion.ingest_file(
                path, source_file, images_root=IMAGES_ROOT, client=client
            )
            for ch in chunks:
                ch.metadata["admin_code"] = DEFAULT_ADMIN_CODE
            print(f"  -> {len(chunks)} chunks  {stats}")
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"  !! failed: {exc}")

    if not all_chunks:
        print("No chunks produced; nothing to write.")
        return

    # Start from a clean store: old vectors may use a different embedding
    # dimension (e.g. the legacy local model), which Chroma cannot mix.
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Removing existing {CHROMA_DB_DIR}/ for a clean rebuild...")
        shutil.rmtree(CHROMA_DB_DIR)

    print(f"Embedding {len(all_chunks)} chunks with {EMBED_MODEL} and writing to {CHROMA_DB_DIR}/ ...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print("Ingestion complete. Vector store saved to disk.")


if __name__ == "__main__":
    main()
