import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEF
from langchain_core.embeddings import Embeddings as _Embeddings


class _LocalEmbeddings(_Embeddings):
    def __init__(self):
        self._ef = _ChromaEF()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._ef(texts)]

    def embed_query(self, text):
        return [float(x) for x in self._ef([text])[0]]
from langchain_community.vectorstores import Chroma

# Resolve path to .env in the same directory as this script
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

# Directories to search
DATASET_DIR = "dataset"
CHROMA_DB_DIR = "chroma_db"

def main():
    print(f"Scanning {DATASET_DIR} for PDF files...")
    # Find all PDFs in all subdirectories
    pdf_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files. Beginning processing...")

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            for doc in docs:
                # Store the filename in metadata
                doc.metadata['source_file'] = os.path.basename(pdf_file)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    print(f"Loaded {len(documents)} pages. Splitting text...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} text chunks. Generating local embeddings and storing in ChromaDB...")

    # Using free, local HuggingFace embeddings (no API key needed!)
    embeddings = _LocalEmbeddings()
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print("Ingestion complete. Vector store saved to disk.")

if __name__ == "__main__":
    main()
