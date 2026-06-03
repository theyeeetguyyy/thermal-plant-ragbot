import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try loading from root (.env) and backend/ (.env)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

app = FastAPI(title="Thermal Power Plant RAG Agent")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Thermal Power Plant RAG API is running."}

# Initialize Global Variables
vectorstore = None
retriever = None
llm = None
embeddings = None

# Query limits tracking
DEMO_LIMIT = 3
demo_usage = {"demo1": 0}

@app.on_event("startup")
async def startup_event():
    global vectorstore, retriever, llm, embeddings
    try:
        # Load API key from env (uses the name 'API')
        api_key = os.environ.get("API")
        if not api_key:
            print("WARNING: API key not set in environment (expected variable name 'API').")
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load ChromaDB
        if os.path.exists(CHROMA_DB_DIR):
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 13})
            
            # Initialize LLM with Groq
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
            
            print("RAG Pipeline initialized successfully.")
        else:
            print("WARNING: Chroma DB directory not found. Please run ingest.py first or upload a document.")
    except Exception as e:
        print(f"Error during startup: {e}")

# Models
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    remaining_queries: int = -1

# Auth Dependency
def get_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    user = parts[1]
    if user not in ["admin", "demo1"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return user

@app.post("/api/login")
async def login(req: LoginRequest):
    if req.username == "admin" and req.password == "admin321":
        return {"token": "admin", "role": "admin"}
    elif req.username == "demo1" and req.password == "demo1":
        return {"token": "demo1", "role": "demo", "remaining": DEMO_LIMIT - demo_usage.get("demo1", 0)}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: str = Depends(get_user)):
    global demo_usage
    
    if user == "demo1":
        if demo_usage.get(user, 0) >= DEMO_LIMIT:
            raise HTTPException(status_code=402, detail="Query limit exceeded. Please upgrade your plan.")
        demo_usage[user] = demo_usage.get(user, 0) + 1

    remaining_queries = -1
    if user == "demo1":
        remaining_queries = DEMO_LIMIT - demo_usage[user]

    if not retriever or not llm:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized. Check if DB exists and API key is set.")
    
    try:
        # 1. Retrieve relevant documents
        docs = retriever.invoke(request.query)
        
        # 2. Extract context and unique sources
        context_parts = []
        sources = []
        for doc in docs:
            context_parts.append(doc.page_content)
            source = doc.metadata.get("source_file", "Unknown")
            if source not in sources:
                sources.append(source)
                
        context = "\n\n".join(context_parts)
        
        # 3. Construct system prompt manually
        system_prompt = (
            "You are an expert assistant for a thermal power plant.\n"
            "Use the following pieces of retrieved context to answer the user's question.\n"
            "If you don't know the answer, say that you don't know based on the provided documents.\n"
            "Always try to cite the source files in your answer.\n\n"
            f"Context:\n{context}"
        )
        
        # 4. Invoke LLM with the custom prompt
        messages = [
            ("system", system_prompt),
            ("human", request.query)
        ]
        
        response = llm.invoke(messages)
        
        return ChatResponse(
            answer=response.content,
            sources=sources,
            remaining_queries=remaining_queries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), user: str = Depends(get_user)):
    if user != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload documents")
    
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
        for doc in docs:
            doc.metadata['source_file'] = file.filename
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(docs)
        
        global vectorstore, retriever, llm, embeddings
        
        if not vectorstore:
            # If no vectorstore yet, create it
            if not embeddings:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 13})
            
            # Init LLM if not done
            if not llm:
                api_key = os.environ.get("API")
                if api_key:
                    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
        else:
            # Add to existing vectorstore
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            
        return {"message": "File uploaded and ingested successfully", "filename": file.filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")
