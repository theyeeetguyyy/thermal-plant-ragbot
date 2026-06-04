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
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from starlette.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

from backend import ingestion

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

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

PERSISTENT_DATA_MOUNT = "/data/v2"
if os.path.exists("/data"):
    os.makedirs(PERSISTENT_DATA_MOUNT, exist_ok=True)
DEMO_LIMIT = int(os.environ.get("DEMO_LIMIT", "10"))
DEMO_UPLOAD_LIMIT = int(os.environ.get("DEMO_UPLOAD_LIMIT", "10"))
CHAT_MEMORY_MESSAGES = int(os.environ.get("CHAT_MEMORY_MESSAGES", "8"))

# ── Administrations (multi-tenant workspaces) ────────────────
# Each administration is a {key, value} pair: code (e.g. "001") + name
# (e.g. "M.P Power Jabalpur"), plus a theme color and an optional logo.
DEFAULT_THEME = "blue"
DEFAULT_ADMIN_CODE = "DEFAULT"
DEFAULT_ADMIN_NAME = "Default Workspace"
LOGO_DIR_NAME = "admin_logos"

# Theme palettes are stored in the DB (db.themes) so they can be tuned or
# extended without code changes. These are the seed defaults. Each palette maps
# directly onto the CSS variables the frontend overrides on :root.
THEME_SEEDS = [
    {
        "key": "blue",
        "name": "Blue",
        "palette": {
            "primary": "#1e5fa8", "primary_dark": "#164a85", "primary_container": "#3b7cc4",
            "primary_subtle": "#d6e7fa", "primary_fixed": "#bcd9f5", "primary_fixed_dim": "#9cc3ec",
            "secondary": "#2f5e8f", "secondary_container": "#cfe1f7",
            "sb_active_bg": "#e6f0fb", "sb_active_text": "#1e5fa8",
        },
    },
    {
        "key": "red",
        "name": "Red",
        "palette": {
            "primary": "#b3261e", "primary_dark": "#8c1d17", "primary_container": "#cf463d",
            "primary_subtle": "#fcdedb", "primary_fixed": "#f7c5c0", "primary_fixed_dim": "#eda6a0",
            "secondary": "#8f2f2a", "secondary_container": "#f7d3cf",
            "sb_active_bg": "#fbe9e7", "sb_active_text": "#b3261e",
        },
    },
    {
        "key": "yellow",
        "name": "Yellow",
        "palette": {
            "primary": "#b8860b", "primary_dark": "#946c08", "primary_container": "#d4a017",
            "primary_subtle": "#fcf3d6", "primary_fixed": "#f5e4a8", "primary_fixed_dim": "#e8d27e",
            "secondary": "#8f6b0a", "secondary_container": "#f7ebc2",
            "sb_active_bg": "#fdf6e3", "sb_active_text": "#946c08",
        },
    },
]

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

if os.path.exists(PERSISTENT_DATA_MOUNT):
    DATASET_DIR = os.path.join(PERSISTENT_DATA_MOUNT, "dataset")
    initial_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(DATASET_DIR) and os.path.exists(initial_dataset):
        shutil.copytree(initial_dataset, DATASET_DIR)
else:
    DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Per-administration dataset folders live under DATASET_DIR/<admin_code>.
# Logos live under LOGO_DIR.
LOGO_DIR = os.path.join(DATASET_DIR, LOGO_DIR_NAME)


def admin_dataset_dir(admin_code: str) -> str:
    path = os.path.join(DATASET_DIR, "admins", admin_code)
    os.makedirs(path, exist_ok=True)
    return path


def backup_chroma():
    if os.path.exists(PERSISTENT_DATA_MOUNT):
        shutil.make_archive(os.path.join(PERSISTENT_DATA_MOUNT, "chroma_backup"), 'zip', CHROMA_DB_DIR)

def restore_chroma():
    if os.path.exists(PERSISTENT_DATA_MOUNT):
        backup_zip = os.path.join(PERSISTENT_DATA_MOUNT, "chroma_backup.zip")
        if os.path.exists(backup_zip):
            if os.path.exists(CHROMA_DB_DIR):
                shutil.rmtree(CHROMA_DB_DIR)
            shutil.unpack_archive(backup_zip, CHROMA_DB_DIR)
        else:
            if os.path.exists(CHROMA_DB_DIR):
                backup_chroma()

mongo_client: AsyncIOMotorClient | None = None
db = None
vectorstore = None
retriever = None
llm = None
embeddings = None
openai_client: OpenAI | None = None


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
    await db.chats.create_index([("admin_code", 1), ("username", 1), ("updated_at", -1)])
    await db.messages.create_index([("chat_id", 1), ("created_at", 1)])
    await db.usage.create_index("username", unique=True)
    # Filenames are now unique *within* an administration, not globally. A
    # legacy DB may still carry the old global `filename_1` unique index, which
    # would wrongly block the same filename living in two administrations — drop
    # it before creating the per-administration compound index.
    try:
        existing = await db.documents.index_information()
        if "filename_1" in existing:
            await db.documents.drop_index("filename_1")
    except Exception as e:
        print(f"Legacy documents index cleanup skipped: {e}")
    await db.documents.create_index([("admin_code", 1), ("filename", 1)], unique=True)
    await db.community_posts.create_index([("admin_code", 1), ("shared_at", -1)])
    await db.administrations.create_index("code", unique=True)
    await db.themes.create_index("key", unique=True)


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
                    "administrations": [DEFAULT_ADMIN_CODE],
                    "active_admin": DEFAULT_ADMIN_CODE,
                }
            },
            upsert=True,
        )


async def seed_themes() -> None:
    """Insert the default theme palettes if they don't already exist."""
    for theme in THEME_SEEDS:
        await db.themes.update_one(
            {"key": theme["key"]},
            {"$setOnInsert": {**theme, "created_at": now_utc()}},
            upsert=True,
        )


async def theme_exists(key: str) -> bool:
    return await db.themes.find_one({"key": key}, {"_id": 1}) is not None


async def seed_administrations() -> None:
    """Create the fallback administration and migrate any pre-existing data."""
    await db.administrations.update_one(
        {"code": DEFAULT_ADMIN_CODE},
        {
            "$setOnInsert": {
                "code": DEFAULT_ADMIN_CODE,
                "name": DEFAULT_ADMIN_NAME,
                "theme": DEFAULT_THEME,
                "logo_path": None,
                "created_at": now_utc(),
            }
        },
        upsert=True,
    )

    # Ensure every existing user belongs to at least the default administration.
    await db.users.update_many(
        {"administrations": {"$exists": False}},
        {"$set": {"administrations": [DEFAULT_ADMIN_CODE], "active_admin": DEFAULT_ADMIN_CODE}},
    )
    await db.users.update_many(
        {"active_admin": {"$exists": False}},
        {"$set": {"active_admin": DEFAULT_ADMIN_CODE}},
    )

    # Tag legacy documents/chats/messages/posts that predate multi-tenancy.
    for coll in (db.documents, db.chats, db.messages, db.community_posts):
        await coll.update_many(
            {"admin_code": {"$exists": False}},
            {"$set": {"admin_code": DEFAULT_ADMIN_CODE}},
        )

    # Tag legacy Chroma chunks so they remain retrievable under DEFAULT.
    try:
        if vectorstore is not None:
            existing = vectorstore._collection.get(include=["metadatas"])
            ids = existing.get("ids", []) or []
            metas = existing.get("metadatas", []) or []
            stale_ids, stale_metas = [], []
            for cid, meta in zip(ids, metas):
                if not meta or "admin_code" not in meta:
                    new_meta = dict(meta or {})
                    new_meta["admin_code"] = DEFAULT_ADMIN_CODE
                    stale_ids.append(cid)
                    stale_metas.append(new_meta)
            if stale_ids:
                vectorstore._collection.update(ids=stale_ids, metadatas=stale_metas)
                print(f"Migrated {len(stale_ids)} legacy Chroma chunks to '{DEFAULT_ADMIN_CODE}'.")
    except Exception as e:
        print(f"Chroma migration skipped: {e}")


async def get_chat_for_user(chat_id: str, username: str, admin_code: str) -> dict[str, Any]:
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    chat = await db.chats.find_one({"_id": ObjectId(chat_id), "username": username, "admin_code": admin_code})
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


async def resolve_active_admin(user: dict[str, Any]) -> str:
    """Return the user's active administration code, repairing it if stale.

    Admins implicitly have access to every administration; regular users are
    confined to the administrations they have been assigned.
    """
    memberships = user.get("administrations") or []
    if user.get("role") == "admin":
        # Admins can act within any administration; default to their last one
        # or the fallback workspace.
        active = user.get("active_admin") or DEFAULT_ADMIN_CODE
        return active
    if not memberships:
        memberships = [DEFAULT_ADMIN_CODE]
        await db.users.update_one(
            {"username": user["username"]},
            {"$set": {"administrations": memberships, "active_admin": DEFAULT_ADMIN_CODE}},
        )
    active = user.get("active_admin")
    if active not in memberships:
        active = memberships[0]
        await db.users.update_one({"username": user["username"]}, {"$set": {"active_admin": active}})
    return active


def serialize_admin(doc: dict[str, Any]) -> dict[str, Any]:
    """Public shape of an administration (key/value + theme + logo flag)."""
    return {
        "code": doc["code"],
        "name": doc.get("name", ""),
        "theme": doc.get("theme", DEFAULT_THEME),
        "has_logo": bool(doc.get("logo_path")),
    }


async def accessible_administrations(user: dict[str, Any]) -> list[dict[str, Any]]:
    """Administrations the user may switch into.

    Admins see every administration; regular users only their assigned ones.
    """
    if user.get("role") == "admin":
        cursor = db.administrations.find().sort("code", 1)
    else:
        codes = user.get("administrations") or [DEFAULT_ADMIN_CODE]
        cursor = db.administrations.find({"code": {"$in": codes}}).sort("code", 1)
    return [serialize_admin(a) async for a in cursor]


async def build_session_context(username: str) -> dict[str, Any]:
    """Administration list + active code + active theme for a login/switch response."""
    user = await db.users.find_one({"username": username})
    active = await resolve_active_admin(user)
    admins = await accessible_administrations(user)
    active_doc = await db.administrations.find_one({"code": active})
    theme = active_doc.get("theme", DEFAULT_THEME) if active_doc else DEFAULT_THEME
    return {"administrations": admins, "active_admin": active, "theme": theme}


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
    admin_code = await resolve_active_admin(user)
    return {
        "username": user["username"],
        "role": user["role"],
        "admin_code": admin_code,
        "administrations": user.get("administrations") or [],
    }


async def require_admin(user: dict[str, Any] = Depends(get_user)) -> dict[str, Any]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


@app.on_event("startup")
async def startup_event():
    global mongo_client, db, vectorstore, retriever, llm, embeddings, openai_client
    restore_chroma()

    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongo_db_name = os.environ.get("MONGODB_DB", "thermal_plant_ragbot")
    mongo_client = AsyncIOMotorClient(mongo_uri)
    db = mongo_client[mongo_db_name]
    await create_indexes()
    await seed_themes()
    await seed_users()

    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set in environment.")

        openai_client = OpenAI(api_key=api_key) if api_key else None
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=api_key)
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=api_key, temperature=0)

        if os.path.exists(CHROMA_DB_DIR):
            try:
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8},
                )
                print("RAG pipeline initialized successfully.")
            except Exception as chroma_err:
                print(f"ChromaDB load failed ({chroma_err}). Wiping stale DB and starting fresh.")
                shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
                vectorstore = None
                retriever = None
        else:
            print("WARNING: Chroma DB directory not found. Upload a document to initialize.")
    except Exception as e:
        print(f"Error during startup: {e}")

    # Runs after the vectorstore is available so legacy Chroma chunks can be
    # tagged with the default administration during migration.
    await seed_administrations()


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

    ctx = await build_session_context(user["username"])
    return {"token": token, "username": user["username"], "role": user["role"], "remaining": remaining, "remaining_uploads": remaining_uploads, **ctx}


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
            "administrations": [DEFAULT_ADMIN_CODE],
            "active_admin": DEFAULT_ADMIN_CODE,
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

    ctx = await build_session_context(req.username)
    return {"token": token, "username": req.username, "role": "demo", "remaining": remaining, "remaining_uploads": remaining_uploads, **ctx}


@app.post("/api/logout")
async def logout(user: dict[str, Any] = Depends(get_user), authorization: str = Header(None)):
    token = authorization.split(" ")[1]
    await db.sessions.delete_one({"token": token, "username": user["username"]})
    return {"message": "Logged out"}


@app.get("/api/chats")
async def list_chats(user: dict[str, Any] = Depends(get_user)):
    chats = await db.chats.find({"username": user["username"], "admin_code": user["admin_code"]}).sort("updated_at", -1).to_list(length=100)
    return {"chats": [serialize_doc(chat) for chat in chats]}


@app.post("/api/chats")
async def create_chat(req: ChatCreateRequest, user: dict[str, Any] = Depends(get_user)):
    result = await db.chats.insert_one(
        {
            "username": user["username"],
            "admin_code": user["admin_code"],
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
    chat = await get_chat_for_user(chat_id, user["username"], user["admin_code"])
    messages = await db.messages.find({"chat_id": chat_id}).sort("created_at", 1).to_list(length=500)
    item = serialize_doc(chat)
    item["messages"] = [serialize_doc(message) for message in messages]
    return item


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user: dict[str, Any] = Depends(get_user)):
    await get_chat_for_user(chat_id, user["username"], user["admin_code"])
    await db.messages.delete_many({"chat_id": chat_id})
    await db.chats.delete_one({"_id": ObjectId(chat_id), "username": user["username"]})
    return {"message": "Chat deleted"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict[str, Any] = Depends(get_user)):
    username = user["username"]
    role = user["role"]
    admin_code = user["admin_code"]

    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check OPENAI_API_KEY setting.")

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
        chat_doc = await get_chat_for_user(chat_id, username, admin_code)
    else:
        result = await db.chats.insert_one(
            {
                "username": username,
                "admin_code": admin_code,
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

        # Fetch the list of ingested documents (scoped to this administration)
        ingested_docs_cursor = db.documents.find({"admin_code": admin_code}, {"filename": 1, "_id": 0})
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
            # Every search is confined to the active administration's chunks.
            if source_file:
                # User pinned a specific document — search only within it, no score cutoff
                is_summary = any(w in request.query.lower() for w in ("summarize", "summary", "overview", "what is", "describe", "explain"))
                k = 20 if is_summary else 12
                chroma_filter = {"$and": [{"admin_code": admin_code}, {"source_file": source_file}]}
            else:
                k = 8
                chroma_filter = {"admin_code": admin_code}
            try:
                raw_docs = vectorstore.similarity_search(request.query, k=k, filter=chroma_filter)
            except Exception as e:
                print(f"Filtered search failed: {e}")
                raw_docs = []

            for i, doc in enumerate(raw_docs, 1):
                text = doc.page_content.strip()
                pdf_refs = text.lower().count(".pdf") + text.lower().count(".docx")
                words = len(text.split())
                if words < 5 or (pdf_refs > 3 and pdf_refs / max(words, 1) > 0.05):
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

            """## HOW TO ANSWER — fit the format to the question, do not force it
- Lead with a direct answer to what was asked, in the first sentence.
- **Highlight the key facts**: bold the specific values, parameters, names or settings that matter, e.g. **drum pressure: 158 kg/cm²(g)** at 100% BMCR.
- Use a **numbered list only when the answer is genuinely a procedure or sequence** of steps; keep conditions, parameters and warnings exactly as stated.
- Use bullets or a small table only when they make the information clearer (e.g. several values, a comparison). For a simple factual question, a short sentence or two is better than an imposed structure.
- Add a brief **Key highlights** recap only for longer or multi-part answers — skip it for short ones.
- Prefix genuine safety-critical notes with ⚠️.
- Note when a value is read from a chart/graph (it is approximate) and cite the page when helpful. Never pad the answer to fill a template.

"""

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
            {"chat_id": chat_id, "admin_code": admin_code, "role": "user", "content": request.query, "created_at": now_utc()}
        )
        await db.messages.insert_one(
            {"chat_id": chat_id, "admin_code": admin_code, "role": "ai", "content": answer, "sources": sources, "created_at": now_utc()}
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
    admin_code = user["admin_code"]
    if user["role"] == "demo":
        usage = await db.usage.find_one({"username": user["username"]}) or {}
        if usage.get("uploads_used", 0) >= DEMO_UPLOAD_LIMIT:
            raise HTTPException(status_code=402, detail=f"Upload limit reached ({DEMO_UPLOAD_LIMIT} files). Contact admin for more.")

    dataset_dir = admin_dataset_dir(admin_code)
    file_path = os.path.join(dataset_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")

        # Dynamic pipeline: PDF -> markdown (+ vision on charts) -> cleanup -> chunks.
        images_root = os.path.join(dataset_dir, "_images")
        chunks, stats = await run_in_threadpool(
            ingestion.ingest_file,
            file_path, file.filename, images_root, openai_client
        )
        for ch in chunks:
            ch.metadata["admin_code"] = admin_code
        print(f"Ingested {file.filename} [{admin_code}]: {stats}")

        if not chunks:
            os.remove(file_path)
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from this file."
            )

        global vectorstore, retriever, llm, embeddings

        if not vectorstore:
            if not embeddings:
                embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=os.environ.get("OPENAI_API_KEY"))
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist()
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8},
            )
            if not llm:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=api_key, temperature=0)
        else:
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            
        backup_chroma()

        await db.documents.update_one(
            {"filename": file.filename, "admin_code": admin_code},
            {
                "$set": {
                    "filename": file.filename,
                    "admin_code": admin_code,
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
    # Documents are isolated per administration. Within an administration,
    # admins see every file; regular users see only their own uploads.
    query: dict[str, Any] = {"admin_code": user["admin_code"]}
    if user["role"] != "admin":
        query["uploaded_by"] = user["username"]
    stored = await db.documents.find(query).sort("uploaded_at", -1).to_list(length=200)
    return {"documents": [serialize_doc(doc) for doc in stored]}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, user: dict[str, Any] = Depends(get_user)):
    admin_code = user["admin_code"]
    doc = await db.documents.find_one({"filename": filename, "admin_code": admin_code})
    if user["role"] != "admin":
        if not doc or doc.get("uploaded_by") != user["username"]:
            raise HTTPException(status_code=403, detail="You can only delete your own documents")

    file_path = doc["path"] if doc and doc.get("path") else os.path.join(admin_dataset_dir(admin_code), filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    global vectorstore
    if vectorstore:
        try:
            # Scope the vector deletion to this administration's copy of the file.
            vectorstore._collection.delete(where={"$and": [{"admin_code": admin_code}, {"source_file": filename}]})
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            raise HTTPException(status_code=500, detail=f"Error removing from vector database: {str(e)}")

        backup_chroma()

    await db.documents.delete_one({"filename": filename, "admin_code": admin_code})
    return {"message": f"{filename} deleted successfully"}


@app.get("/api/documents/{filename}/view")
async def view_document(filename: str, token: str | None = None):
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    session = await db.sessions.find_one({"token": token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await db.users.find_one({"username": session["username"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    admin_code = await resolve_active_admin(user)

    doc = await db.documents.find_one({"filename": filename, "admin_code": admin_code})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in this administration")
    file_path = doc["path"] if doc.get("path") else os.path.join(admin_dataset_dir(admin_code), filename)

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
    doc_cursor = db.documents.find({"admin_code": user["admin_code"]}, {"filename": 1, "_id": 0})
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
        "admin_code": user["admin_code"],
        "shared_at": now_utc(),
    })
    post = await db.community_posts.find_one({"_id": result.inserted_id})
    return serialize_doc(post)


@app.get("/api/community")
async def list_community(user: dict[str, Any] = Depends(get_user)):
    posts = await db.community_posts.find({"admin_code": user["admin_code"]}).sort("shared_at", -1).to_list(length=200)
    return {"posts": [serialize_doc(p) for p in posts]}


@app.delete("/api/community/{post_id}")
async def delete_community_post(post_id: str, user: dict[str, Any] = Depends(get_user)):
    if not ObjectId.is_valid(post_id):
        raise HTTPException(status_code=404, detail="Post not found")
    post = await db.community_posts.find_one({"_id": ObjectId(post_id), "admin_code": user["admin_code"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if user["role"] != "admin" and post.get("shared_by") != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to delete this post")
    await db.community_posts.delete_one({"_id": ObjectId(post_id)})
    return {"message": "Post deleted"}


# ════════════════════════════════════════════════════════════
#  Administrations (multi-tenant workspaces) + themes
# ════════════════════════════════════════════════════════════

class AdminCreateRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=120)
    theme: str = DEFAULT_THEME


class AdminUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    theme: str | None = None


class MemberRequest(BaseModel):
    username: str = Field(..., min_length=1)


class SwitchAdminRequest(BaseModel):
    code: str = Field(..., min_length=1)


@app.get("/api/themes")
async def list_themes(user: dict[str, Any] = Depends(get_user)):
    """All theme palettes, so the frontend can apply colors from the DB."""
    themes = await db.themes.find({}, {"_id": 0}).sort("key", 1).to_list(length=50)
    return {"themes": themes}


@app.get("/api/me/administrations")
async def my_administrations(user: dict[str, Any] = Depends(get_user)):
    """Administrations the current user can switch into + their active one."""
    return await build_session_context(user["username"])


@app.post("/api/me/active-admin")
async def switch_active_admin(req: SwitchAdminRequest, user: dict[str, Any] = Depends(get_user)):
    """Switch the active administration (persisted, so it stays the default)."""
    target = await db.administrations.find_one({"code": req.code})
    if not target:
        raise HTTPException(status_code=404, detail="Administration not found")
    if user["role"] != "admin" and req.code not in (user.get("administrations") or []):
        raise HTTPException(status_code=403, detail="You don't have access to this administration")
    await db.users.update_one({"username": user["username"]}, {"$set": {"active_admin": req.code}})
    return await build_session_context(user["username"])


@app.get("/api/administrations")
async def admin_list_administrations(user: dict[str, Any] = Depends(require_admin)):
    """Full administration listing with member usernames (admin only)."""
    admins = await db.administrations.find().sort("code", 1).to_list(length=500)
    out = []
    for a in admins:
        members = await db.users.find({"administrations": a["code"]}, {"username": 1, "_id": 0}).to_list(length=500)
        item = serialize_admin(a)
        item["members"] = [m["username"] for m in members]
        out.append(item)
    return {"administrations": out}


@app.post("/api/administrations")
async def create_administration(req: AdminCreateRequest, user: dict[str, Any] = Depends(require_admin)):
    if not await theme_exists(req.theme):
        raise HTTPException(status_code=400, detail="Unknown theme")
    if await db.administrations.find_one({"code": req.code}):
        raise HTTPException(status_code=409, detail="An administration with this code already exists")
    await db.administrations.insert_one({
        "code": req.code,
        "name": req.name,
        "theme": req.theme,
        "logo_path": None,
        "created_at": now_utc(),
    })
    doc = await db.administrations.find_one({"code": req.code})
    return serialize_admin(doc)


@app.patch("/api/administrations/{code}")
async def update_administration(code: str, req: AdminUpdateRequest, user: dict[str, Any] = Depends(require_admin)):
    doc = await db.administrations.find_one({"code": code})
    if not doc:
        raise HTTPException(status_code=404, detail="Administration not found")
    changes: dict[str, Any] = {}
    if req.name is not None:
        changes["name"] = req.name
    if req.theme is not None:
        if not await theme_exists(req.theme):
            raise HTTPException(status_code=400, detail="Unknown theme")
        changes["theme"] = req.theme
    if changes:
        await db.administrations.update_one({"code": code}, {"$set": changes})
    return serialize_admin(await db.administrations.find_one({"code": code}))


@app.delete("/api/administrations/{code}")
async def delete_administration(code: str, user: dict[str, Any] = Depends(require_admin)):
    if code == DEFAULT_ADMIN_CODE:
        raise HTTPException(status_code=400, detail="The default administration cannot be deleted")
    doc = await db.administrations.find_one({"code": code})
    if not doc:
        raise HTTPException(status_code=404, detail="Administration not found")
    # Detach members and reset anyone whose active workspace was this one.
    await db.users.update_many({"administrations": code}, {"$pull": {"administrations": code}})
    await db.users.update_many({"active_admin": code}, {"$set": {"active_admin": DEFAULT_ADMIN_CODE}})
    if doc.get("logo_path") and os.path.exists(doc["logo_path"]):
        try:
            os.remove(doc["logo_path"])
        except OSError:
            pass
    await db.administrations.delete_one({"code": code})
    return {"message": f"Administration {code} deleted"}


@app.post("/api/administrations/{code}/members")
async def add_member(code: str, req: MemberRequest, user: dict[str, Any] = Depends(require_admin)):
    if not await db.administrations.find_one({"code": code}):
        raise HTTPException(status_code=404, detail="Administration not found")
    target = await db.users.find_one({"username": req.username})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    await db.users.update_one({"username": req.username}, {"$addToSet": {"administrations": code}})
    return {"message": f"{req.username} added to {code}"}


@app.delete("/api/administrations/{code}/members/{username}")
async def remove_member(code: str, username: str, user: dict[str, Any] = Depends(require_admin)):
    await db.users.update_one({"username": username}, {"$pull": {"administrations": code}})
    # If they were sitting in this administration, send them back to default.
    await db.users.update_one(
        {"username": username, "active_admin": code},
        {"$set": {"active_admin": DEFAULT_ADMIN_CODE}},
    )
    return {"message": f"{username} removed from {code}"}


@app.get("/api/admin/users")
async def admin_list_users(user: dict[str, Any] = Depends(require_admin)):
    """All users + their administration memberships (for assignment UI)."""
    users = await db.users.find({}, {"username": 1, "role": 1, "administrations": 1, "_id": 0}).sort("username", 1).to_list(length=1000)
    for u in users:
        u["administrations"] = u.get("administrations") or []
    return {"users": users}


@app.post("/api/administrations/{code}/logo")
async def upload_admin_logo(code: str, file: UploadFile = File(...), user: dict[str, Any] = Depends(require_admin)):
    if not await db.administrations.find_one({"code": code}):
        raise HTTPException(status_code=404, detail="Administration not found")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"):
        raise HTTPException(status_code=400, detail="Logo must be PNG, JPG, WEBP, SVG or GIF")
    os.makedirs(LOGO_DIR, exist_ok=True)
    logo_path = os.path.join(LOGO_DIR, f"{code}{ext}")
    # Remove any previous logo with a different extension.
    for old in os.listdir(LOGO_DIR) if os.path.exists(LOGO_DIR) else []:
        if os.path.splitext(old)[0] == code and old != os.path.basename(logo_path):
            try:
                os.remove(os.path.join(LOGO_DIR, old))
            except OSError:
                pass
    with open(logo_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await db.administrations.update_one({"code": code}, {"$set": {"logo_path": logo_path}})
    return {"message": "Logo updated", "has_logo": True}


@app.get("/api/administrations/{code}/logo")
async def get_admin_logo(code: str):
    """Public logo endpoint (no auth) so it can be used in <img> tags."""
    doc = await db.administrations.find_one({"code": code})
    if not doc or not doc.get("logo_path") or not os.path.exists(doc["logo_path"]):
        raise HTTPException(status_code=404, detail="No logo set")
    ext = os.path.splitext(doc["logo_path"])[1].lower()
    media = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".svg": "image/svg+xml", ".gif": "image/gif",
    }.get(ext, "application/octet-stream")
    return FileResponse(doc["logo_path"], media_type=media)
