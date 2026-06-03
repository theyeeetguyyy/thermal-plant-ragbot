import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Thermal Power Plant RAG API is running."}

# Initialize Global Variables
vectorstore = None
retriever = None
llm = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, retriever, llm
    try:
        # Load API key from env (uses the name 'API')
        api_key = os.environ.get("API")
        if not api_key:
            print("WARNING: API key not set in environment (expected variable name 'API').")
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load ChromaDB
        if os.path.exists(CHROMA_DB_DIR):
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k":10}
)
            
            # Initialize LLM with Groq
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
            
            print("RAG Pipeline initialized successfully.")
        else:
            print("WARNING: Chroma DB directory not found. Please run ingest.py first.")
    except Exception as e:
        print(f"Error during startup: {e}")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
        
        # 4. Invoke LLM with the custom prompt (using system / user message format)
        messages = [
            ("system", system_prompt),
            ("human", request.query)
        ]
        
        response = llm.invoke(messages)
        
        return ChatResponse(
            answer=response.content,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
