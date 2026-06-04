import hashlib
import os
import site
import shutil
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

user_site = site.getusersitepackages()
workspace_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".python_packages"))
if os.path.isdir(workspace_packages) and workspace_packages not in sys.path:
    sys.path.append(workspace_packages)
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEF
from langchain_core.embeddings import Embeddings as _Embeddings


class _LocalEmbeddings(_Embeddings):
    """Thin LangChain wrapper around chromadb's built-in ONNX embedding (no DLL issues)."""
    def __init__(self):
        self._ef = _ChromaEF()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._ef([text])[0]]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = FastAPI(title="Thermal Power Plant RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSISTENT_DATA_MOUNT = "/data"
DEMO_LIMIT = int(os.environ.get("DEMO_LIMIT", "10"))
DEMO_UPLOAD_LIMIT = int(os.environ.get("DEMO_UPLOAD_LIMIT", "10"))
CHAT_MEMORY_MESSAGES = int(os.environ.get("CHAT_MEMORY_MESSAGES", "8"))

if os.path.exists(PERSISTENT_DATA_MOUNT):
    CHROMA_DB_DIR = os.path.join(PERSISTENT_DATA_MOUNT, "chroma_db")
    DATASET_DIR = os.path.join(PERSISTENT_DATA_MOUNT, "dataset")
    initial_chroma = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
    initial_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(CHROMA_DB_DIR) and os.path.exists(initial_chroma):
        shutil.copytree(initial_chroma, CHROMA_DB_DIR)
    if not os.path.exists(DATASET_DIR) and os.path.exists(initial_dataset):
        shutil.copytree(initial_dataset, DATASET_DIR)
else:
    CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
    DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

mongo_client: AsyncIOMotorClient | None = None
db = None
vectorstore = None
retriever = None
llm = None
embeddings = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    salt = os.environ.get("PASSWORD_SALT", "thermal-plant-ragbot")
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def serialize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    item = dict(doc)
    item["id"] = str(item.pop("_id"))
    for key in ("created_at", "updated_at", "last_message_at", "uploaded_at"):
        if item.get(key):
            item[key] = item[key].isoformat()
    return item


async def create_indexes() -> None:
    await db.users.create_index("username", unique=True)
    await db.sessions.create_index("token", unique=True)
    await db.sessions.create_index("username")
    await db.chats.create_index([("username", 1), ("updated_at", -1)])
    await db.messages.create_index([("chat_id", 1), ("created_at", 1)])
    await db.usage.create_index("username", unique=True)
    await db.documents.create_index("filename", unique=True)
    await db.community_posts.create_index([("shared_at", -1)])


async def seed_users() -> None:
    seeds = [
        {"username": "admin", "password": "admin321", "role": "admin"},
        {"username": "demo1", "password": "demo1", "role": "demo"},
    ]
    for user in seeds:
        await db.users.update_one(
            {"username": user["username"]},
            {
                "$setOnInsert": {
                    "username": user["username"],
                    "password_hash": hash_password(user["password"]),
                    "role": user["role"],
                    "created_at": now_utc(),
                }
            },
            upsert=True,
        )


async def get_chat_for_user(chat_id: str, username: str) -> dict[str, Any]:
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    chat = await db.chats.find_one({"_id": ObjectId(chat_id), "username": username})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


async def get_recent_chat_memory(chat_id: str) -> str:
    cursor = (
        db.messages.find({"chat_id": chat_id}, {"role": 1, "content": 1, "_id": 0})
        .sort("created_at", -1)
        .limit(CHAT_MEMORY_MESSAGES)
    )
    messages = list(reversed(await cursor.to_list(length=CHAT_MEMORY_MESSAGES)))
    if not messages:
        return "No prior messages in this chat."
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=6)


class ChatCreateRequest(BaseModel):
    title: str = "New Chat"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chat_id: str | None = None
    source_file: str | None = None


class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    sources: list[str]
    remaining_queries: int = -1


class SharePostRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    sources: list[str] = []


async def get_user(authorization: str = Header(None)) -> dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    session = await db.sessions.find_one({"token": parts[1]})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    user = await db.users.find_one({"username": session["username"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"username": user["username"], "role": user["role"]}


@app.on_event("startup")
async def startup_event():
    global mongo_client, db, vectorstore, retriever, llm, embeddings

    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongo_db_name = os.environ.get("MONGODB_DB", "thermal_plant_ragbot")
    mongo_client = AsyncIOMotorClient(mongo_uri)
    db = mongo_client[mongo_db_name]
    await create_indexes()
    await seed_users()

    try:
        api_key = os.environ.get("API")
        if not api_key:
            print("WARNING: API key not set in environment (expected variable name 'API').")

        embeddings = _LocalEmbeddings()

        if os.path.exists(CHROMA_DB_DIR):
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 8, "score_threshold": 0.25},
            )
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
            print("RAG pipeline initialized successfully.")
        else:
            print("WARNING: Chroma DB directory not found. Run ingest.py first or upload a document.")
    except Exception as e:
        print(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    if mongo_client:
        mongo_client.close()


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Thermal Power Plant RAG API is running."}


@app.post("/api/login")
async def login(req: LoginRequest):
    user = await db.users.find_one({"username": req.username})
    if not user or user["password_hash"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    await db.sessions.insert_one(
        {
            "token": token,
            "username": user["username"],
            "created_at": now_utc(),
            "last_seen_at": now_utc(),
        }
    )

    remaining = -1
    remaining_uploads = -1
    if user["role"] == "demo":
        usage = await db.usage.find_one({"username": user["username"]}) or {}
        remaining = max(DEMO_LIMIT - usage.get("queries_used", 0), 0)
        remaining_uploads = max(DEMO_UPLOAD_LIMIT - usage.get("uploads_used", 0), 0)

    return {"token": token, "username": user["username"], "role": user["role"], "remaining": remaining, "remaining_uploads": remaining_uploads}


@app.post("/api/register")
async def register(req: RegisterRequest):
    existing = await db.users.find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    await db.users.insert_one(
        {
            "username": req.username,
            "password_hash": hash_password(req.password),
            "role": "demo",
            "created_at": now_utc(),
        }
    )

    token = str(uuid.uuid4())
    await db.sessions.insert_one(
        {
            "token": token,
            "username": req.username,
            "created_at": now_utc(),
            "last_seen_at": now_utc(),
        }
    )

    usage = await db.usage.find_one({"username": req.username}) or {}
    remaining = max(DEMO_LIMIT - usage.get("queries_used", 0), 0)
    remaining_uploads = max(DEMO_UPLOAD_LIMIT - usage.get("uploads_used", 0), 0)

    return {"token": token, "username": req.username, "role": "demo", "remaining": remaining, "remaining_uploads": remaining_uploads}


@app.post("/api/logout")
async def logout(user: dict[str, Any] = Depends(get_user), authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    await db.sessions.delete_one({"token": token, "username": user["username"]})
    return {"message": "Logged out"}


@app.get("/api/chats")
async def list_chats(user: dict[str, Any] = Depends(get_user)):
    chats = await db.chats.find({"username": user["username"]}).sort("updated_at", -1).to_list(length=100)
    return {"chats": [serialize_doc(chat) for chat in chats]}


@app.post("/api/chats")
async def create_chat(req: ChatCreateRequest, user: dict[str, Any] = Depends(get_user)):
    result = await db.chats.insert_one(
        {
            "username": user["username"],
            "title": req.title[:80] or "New Chat",
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "last_message_at": None,
        }
    )
    chat = await db.chats.find_one({"_id": result.inserted_id})
    return serialize_doc(chat)


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, user: dict[str, Any] = Depends(get_user)):
    chat = await get_chat_for_user(chat_id, user["username"])
    messages = await db.messages.find({"chat_id": chat_id}).sort("created_at", 1).to_list(length=500)
    item = serialize_doc(chat)
    item["messages"] = [serialize_doc(message) for message in messages]
    return item


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user: dict[str, Any] = Depends(get_user)):
    await get_chat_for_user(chat_id, user["username"])
    await db.messages.delete_many({"chat_id": chat_id})
    await db.chats.delete_one({"_id": ObjectId(chat_id), "username": user["username"]})
    return {"message": "Chat deleted"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict[str, Any] = Depends(get_user)):
    username = user["username"]
    role = user["role"]

    if not retriever or not llm:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized. Check DB and API key settings.")

    if role == "demo":
        usage = await db.usage.find_one({"username": username}) or {"queries_used": 0}
        if usage.get("queries_used", 0) >= DEMO_LIMIT:
            raise HTTPException(status_code=402, detail="Query limit exceeded. Please upgrade your plan.")
        await db.usage.update_one(
            {"username": username},
            {"$inc": {"queries_used": 1}, "$set": {"updated_at": now_utc()}, "$setOnInsert": {"created_at": now_utc()}},
            upsert=True,
        )

    remaining_queries = -1
    if role == "demo":
        usage = await db.usage.find_one({"username": username}) or {"queries_used": 0}
        remaining_queries = max(DEMO_LIMIT - usage.get("queries_used", 0), 0)

    chat_id = request.chat_id
    if chat_id:
        chat_doc = await get_chat_for_user(chat_id, username)
    else:
        result = await db.chats.insert_one(
            {
                "username": username,
                "title": request.query[:60],
                "created_at": now_utc(),
                "updated_at": now_utc(),
                "last_message_at": None,
            }
        )
        chat_id = str(result.inserted_id)
        chat_doc = await db.chats.find_one({"_id": result.inserted_id})

    try:
        chat_memory = await get_recent_chat_memory(chat_id)

        # Fetch the list of ingested documents for meta-question awareness
        ingested_docs_cursor = db.documents.find({}, {"filename": 1, "_id": 0})
        ingested_docs_list = [d["filename"] async for d in ingested_docs_cursor]
        doc_list_text = (
            f"{len(ingested_docs_list)} document(s) ingested: " + ", ".join(ingested_docs_list)
            if ingested_docs_list
            else "No documents have been uploaded yet."
        )

        source_file = request.source_file
        sources = []
        context_parts = []

        if vectorstore:
            if source_file:
                # User pinned a specific document — search only within it, no score cutoff
                try:
                    is_summary = any(w in request.query.lower() for w in ("summarize", "summary", "overview", "what is", "describe", "explain"))
                    k = 20 if is_summary else 12
                    raw_docs = vectorstore.similarity_search(
                        request.query,
                        k=k,
                        filter={"source_file": source_file},
                    )
                except Exception as e:
                    print(f"Filtered search failed, falling back: {e}")
                    raw_docs = retriever.invoke(request.query)
            else:
                raw_docs = retriever.invoke(request.query)

            for i, doc in enumerate(raw_docs, 1):
                text = doc.page_content.strip()
                pdf_refs = text.lower().count(".pdf") + text.lower().count(".docx")
                words = len(text.split())
                if words < 15 or (pdf_refs > 3 and pdf_refs / max(words, 1) > 0.05):
                    continue
                src = doc.metadata.get("source_file") or os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page")
                page_label = f" (page {page + 1})" if page is not None else ""
                context_parts.append(f"[CHUNK {i} | Source: {src}{page_label}]\n{text}")
                if src not in sources:
                    sources.append(src)

        retrieved_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # Build scoped-document note when user has pinned a file
        scoped_note = (
            f"## Active document scope\n"
            f"The user has selected **{source_file}** as the active document for this session. "
            f"Base your answer primarily on content from this document. "
            f"If asked to summarise, provide a comprehensive summary of the retrieved chunks below.\n\n"
        ) if source_file else ""

        no_context_msg = (
            f"The document **{source_file}** was selected but no matching content could be retrieved for this query. "
            f"Try rephrasing, or ask a broader question about the document."
        ) if source_file else (
            "The uploaded documents do not contain information on this topic. "
            "Please upload the relevant manual or SOP."
        )

        system_prompt = (
            "You are a document-grounded assistant for a thermal power plant operations team.\n\n"

            "## STRICT RULES — follow these without exception\n"
            "1. Answer ONLY using information found in the RETRIEVED DOCUMENT CONTEXT below.\n"
            "2. Do NOT use your general training knowledge, background knowledge, or assumptions.\n"
            f"3. If the answer is not in the retrieved context, say exactly: \"{no_context_msg}\"\n"
            "4. Never invent facts, procedures, values, or specifications.\n\n"

            + scoped_note +

            "## OUTPUT FORMAT — always structure your answer like this\n"
            "- **Procedures / steps**: Use a numbered list (1. 2. 3. …). One action per step. "
            "Include any conditions, parameters, or warnings exactly as stated in the document.\n"
            "- **Key features / components**: Use a bullet list with bold labels, e.g. `- **Boiler drum**: …`\n"
            "- **Process flows**: Show as a numbered sequence with → arrows between stages where helpful.\n"
            "- **Specifications / values**: Present in a small table or inline bold: `**Pressure**: 150 bar`\n"
            "- **Warnings / safety notes**: Prefix with ⚠️ and put them before the relevant step.\n\n"

            "## Casual / meta questions\n"
            "- Greetings: reply briefly (1 sentence).\n"
            "- Questions about available documents: answer from the DOCUMENT LIBRARY list below.\n\n"

            f"## Document library\n{doc_list_text}\n\n"
            f"## Conversation history (this chat)\n{chat_memory}\n\n"
            + (
                f"## Retrieved document context\n{retrieved_context}"
                if retrieved_context
                else f"## Retrieved document context\nNo relevant content was found for this query."
            )
        )

        response = llm.invoke([("system", system_prompt), ("human", request.query)])
        answer = response.content

        await db.messages.insert_one(
            {"chat_id": chat_id, "role": "user", "content": request.query, "created_at": now_utc()}
        )
        await db.messages.insert_one(
            {"chat_id": chat_id, "role": "ai", "content": answer, "sources": sources, "created_at": now_utc()}
        )

        title = chat_doc.get("title") or "New Chat"
        if title == "New Chat":
            title = request.query[:60]
        await db.chats.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"title": title, "updated_at": now_utc(), "last_message_at": now_utc()}},
        )

        return ChatResponse(
            chat_id=chat_id,
            answer=answer,
            sources=sources,
            remaining_queries=remaining_queries,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), user: dict[str, Any] = Depends(get_user)):
    if user["role"] == "demo":
        usage = await db.usage.find_one({"username": user["username"]}) or {}
        if usage.get("uploads_used", 0) >= DEMO_UPLOAD_LIMIT:
            raise HTTPException(status_code=402, detail=f"Upload limit reached ({DEMO_UPLOAD_LIMIT} files). Contact admin for more.")

    os.makedirs(DATASET_DIR, exist_ok=True)
    file_path = os.path.join(DATASET_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")

        docs = loader.load()
        # Filter out pages with no text (scanned/image pages)
        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        for doc in docs:
            doc.metadata["source_file"] = file.filename

        if not docs:
            os.remove(file_path)
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this file. It may be a scanned/image-only PDF. Please use a text-based PDF or DOCX."
            )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            os.remove(file_path)
            raise HTTPException(status_code=422, detail="Document appears to be empty after processing.")

        global vectorstore, retriever, llm, embeddings

        if not vectorstore:
            if not embeddings:
                embeddings = _LocalEmbeddings()
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist()
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 8, "score_threshold": 0.25},
            )
            if not llm:
                api_key = os.environ.get("API")
                if api_key:
                    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
        else:
            vectorstore.add_documents(chunks)
            vectorstore.persist()

        await db.documents.update_one(
            {"filename": file.filename},
            {
                "$set": {
                    "filename": file.filename,
                    "path": file_path,
                    "uploaded_by": user["username"],
                    "uploaded_at": now_utc(),
                    "chunk_count": len(chunks),
                    "content_type": file.content_type,
                }
            },
            upsert=True,
        )

        if user["role"] == "demo":
            await db.usage.update_one(
                {"username": user["username"]},
                {"$inc": {"uploads_used": 1}, "$setOnInsert": {"created_at": now_utc()}},
                upsert=True,
            )
            usage = await db.usage.find_one({"username": user["username"]}) or {}
            remaining_uploads = max(DEMO_UPLOAD_LIMIT - usage.get("uploads_used", 0), 0)
            return {"message": "File uploaded and ingested successfully", "filename": file.filename, "remaining_uploads": remaining_uploads}

        return {"message": "File uploaded and ingested successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")


@app.get("/api/documents")
async def list_documents(user: dict[str, Any] = Depends(get_user)):
    query = {} if user["role"] == "admin" else {"uploaded_by": user["username"]}
    stored = await db.documents.find(query).sort("uploaded_at", -1).to_list(length=200)
    if stored:
        return {"documents": [serialize_doc(doc) for doc in stored]}

    if user["role"] != "admin":
        return {"documents": []}

    if not os.path.exists(DATASET_DIR):
        return {"documents": []}

    files = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
    return {"documents": [{"filename": f} for f in files]}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, user: dict[str, Any] = Depends(get_user)):
    doc = await db.documents.find_one({"filename": filename})
    if user["role"] != "admin":
        if not doc or doc.get("uploaded_by") != user["username"]:
            raise HTTPException(status_code=403, detail="You can only delete your own documents")

    file_path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    global vectorstore
    if vectorstore:
        try:
            vectorstore._collection.delete(where={"source_file": filename})
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            raise HTTPException(status_code=500, detail=f"Error removing from vector database: {str(e)}")

    await db.documents.delete_one({"filename": filename})
    return {"message": f"{filename} deleted successfully"}


@app.get("/api/documents/{filename}/view")
async def view_document(filename: str, token: str | None = None):
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    session = await db.sessions.find_one({"token": token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    doc = await db.documents.find_one({"filename": filename})
    file_path = doc["path"] if doc and doc.get("path") else os.path.join(DATASET_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    safe_path = os.path.realpath(file_path)
    safe_base = os.path.realpath(DATASET_DIR)
    if not safe_path.startswith(safe_base):
        raise HTTPException(status_code=403, detail="Access denied")

    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(safe_path, media_type=media_type, filename=filename)


@app.get("/api/suggestions")
async def get_suggestions(user: dict[str, Any] = Depends(get_user)):
    doc_cursor = db.documents.find({}, {"filename": 1, "_id": 0})
    doc_names = [d["filename"] async for d in doc_cursor]

    if not doc_names or not llm:
        return {"suggestions": [
            "What documents are available?",
            "Give me an overview of the uploaded materials.",
            "What topics are covered in the documents?",
        ]}

    names_str = ", ".join(doc_names)
    prompt = (
        f"The following documents are uploaded in a thermal power plant knowledge assistant: {names_str}.\n\n"
        "Generate exactly 3 short, specific, useful questions a plant engineer might ask based on these document names. "
        "Each question should be practical and directly answerable from such documents. "
        "Return ONLY a JSON array of 3 strings, nothing else. Example: [\"Q1\", \"Q2\", \"Q3\"]"
    )
    try:
        resp = llm.invoke([("user", prompt)])
        import json, re
        match = re.search(r'\[.*?\]', resp.content, re.DOTALL)
        questions = json.loads(match.group()) if match else []
        if len(questions) == 3 and all(isinstance(q, str) for q in questions):
            return {"suggestions": questions}
    except Exception:
        pass

    return {"suggestions": [
        f"Summarize the key points in {doc_names[0]}",
        "What safety procedures are mentioned in the documents?",
        "What are the main operational guidelines?",
    ]}


@app.post("/api/community")
async def share_to_community(req: SharePostRequest, user: dict[str, Any] = Depends(get_user)):
    result = await db.community_posts.insert_one({
        "question": req.question,
        "answer": req.answer,
        "sources": req.sources,
        "shared_by": user["username"],
        "shared_at": now_utc(),
    })
    post = await db.community_posts.find_one({"_id": result.inserted_id})
    return serialize_doc(post)


@app.get("/api/community")
async def list_community(user: dict[str, Any] = Depends(get_user)):
    posts = await db.community_posts.find().sort("shared_at", -1).to_list(length=200)
    return {"posts": [serialize_doc(p) for p in posts]}


@app.delete("/api/community/{post_id}")
async def delete_community_post(post_id: str, user: dict[str, Any] = Depends(get_user)):
    if not ObjectId.is_valid(post_id):
        raise HTTPException(status_code=404, detail="Post not found")
    post = await db.community_posts.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if user["role"] != "admin" and post.get("shared_by") != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to delete this post")
    await db.community_posts.delete_one({"_id": ObjectId(post_id)})
    return {"message": "Post deleted"}
