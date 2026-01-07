"""
Agentic RAG Backend with Groq API + Tavily Web Search
FastAPI server that classifies a query into two paths:
1. Web Search (Tavily)   -> ONLY for live news/events.
2. General LLM           -> For everything else (Logic, Math, General Knowledge).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import asyncio # Added for non-blocking web requests

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# LangChain & Tavily Integrations
# ------------------------------------------------------------------------------

try:
    from tavily import TavilyClient
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.documents import Document
    LANGCHAIN_DEPS_AVAILABLE = True
except ImportError:
    LANGCHAIN_DEPS_AVAILABLE = False
    print("Warning: Install dependencies with: pip install tavily-python langchain-openai langchain-community langchain-chroma sentence-transformers")

# ------------------------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------------------------

app = FastAPI(title="Agentic RAG API with Groq & Tavily")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

class ChatMessage(BaseModel):
    type: str  # "user", "agent", "thought"
    content: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    final_answer: str

# ------------------------------------------------------------------------------
# Database: Vector (Chroma) - Loads data.txt
# ------------------------------------------------------------------------------

def init_vector_db():
    DATA_FILE = "data.txt"
    
    if os.path.exists(DATA_FILE):
        print(f"Loading documents from '{DATA_FILE}'...")
        loader = TextLoader(DATA_FILE, encoding="utf-8")
        docs = loader.load()
    else:
        print(f"Warning: '{DATA_FILE}' not found. Creating dummy data.")
        docs = [Document(page_content="No data.txt found. Please add data.txt to the backend folder.")]

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    
    # Fix file path syntax
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(script_dir, "chroma_db_standard")
    
    # Store
    store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        collection_name="standard_rag_groq", 
        persist_directory=db_dir
    )
    
    # Return retriever (k=5 for broader context)
    return store.as_retriever(search_kwargs={"k": 5})

# Initialize Global Retriever
vector_retriever = init_vector_db()

# ------------------------------------------------------------------------------
# External Tool Setup (Tavily & Groq)
# ------------------------------------------------------------------------------

# Initialize Tavily
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if not tavily_api_key:
    print("WARNING: TAVILY_API_KEY not found in env. Web search will fail.")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=tavily_api_key)

def get_llm(temperature: float = 0.0):
    api_key = os.environ.get("GROQ_API_KEY") 
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

# ------------------------------------------------------------------------------
# Core "Agent" Functions
# ------------------------------------------------------------------------------

async def classify_query(query: str) -> str:
    """
    Classifies the query into WEB_QUERY or GENERAL_QUERY.
    Prioritizes GENERAL_QUERY unless live external data is strictly required.
    """
    llm = get_llm(temperature=0.0)
    
    # *** UPDATED PROMPT: REMOVED PRODUCT, ADDED LAZY WEB LOGIC ***
    prompt = f"""You are a "Query Routing Agent". Your goal is to minimize web searches.
Classify the user's query into exactly one of these two categories:

Categories:
1. **WEB_QUERY**: Select this ONLY if the user asks for:
   - Recent news (last 24-48 months)
   - Current events (sports scores, stock prices, weather)
   - Specific real-time data that an AI model trained in the past would not know.
   
2. **GENERAL_QUERY**: Select this for EVERYTHING else, including:
   - General knowledge (history, science, facts)
   - Logic, math, and coding
   - Chitchat and greetings
   - Explanations of concepts
   - Hypothetical scenarios

User Query: "{query}"

Respond with ONLY the category name.
"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    classification = response.content.strip().upper()
    
    # Simple fallback logic
    if "WEB" in classification:
        return "WEB_QUERY"
    else:
        return "GENERAL_QUERY"

# --- RAG Functions ---
async def simple_rag_search(query: str) -> str:
    if not vector_retriever: return "Database not initialized."
    docs = await vector_retriever.ainvoke(query)
    if not docs: return "No relevant documents found."
    return "\n\n---\n\n".join([d.page_content for d in docs])

async def generate_rag_answer(query: str, context: str) -> str:
    llm = get_llm(temperature=0.3)
    prompt = f"""You are a Store Assistant. Answer using ONLY this context:\n{context}\n\nQuestion: {query}"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content

# --- Web Search Functions (Tavily) ---
async def perform_web_search(query: str) -> str:
    """
    Runs the search using Tavily and formats the results.
    """
    if not tavily_client:
        return "Error: Tavily API Key is missing."

    print(f"Searching Tavily for: {query}")
    
    # Run in thread pool to prevent blocking asyncio loop
    try:
        response = await asyncio.to_thread(tavily_client.search, query=query, search_depth="basic")
    except Exception as e:
        return f"Error connecting to search engine: {e}"

    # Format results
    results = response.get('results', [])
    if not results:
        return "No search results found."
        
    formatted_context = "Web Search Results:\n"
    for item in results[:3]: # Take top 3 results
        formatted_context += f"- Title: {item['title']}\n  Content: {item['content']}\n  URL: {item['url']}\n\n"
        
    return formatted_context

async def generate_web_answer(query: str, context: str) -> str:
    """Generates an answer based on web search results."""
    llm = get_llm(temperature=0.5)
    
    prompt = f"""You are a Research Assistant. 
Answer the user's question based on the web search results provided below.
If the results don't answer the question, admit it.

Context:
---
{context}
---

User Question: {query}

Answer:"""
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content

# --- General Functions ---
async def generate_general_answer(query: str) -> str:
    llm = get_llm(temperature=0.7)
    prompt = f"""You are a helpful assistant. Answer the user: {query}"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "llama-3.1-8b-instant"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: QueryRequest):
    messages: List[ChatMessage] = []
    try:
        # 1. Classification
        messages.append(ChatMessage(type="thought", content="🤔 Agent is classifying query...", timestamp=datetime.now().isoformat()))
        classification = await classify_query(req.query)
        messages.append(ChatMessage(type="thought", content=f"🏷️ Decision: **{classification}** pipeline.", timestamp=datetime.now().isoformat()))

        final = ""

        # *** UPDATED ROUTING LOGIC ***
        # PATH 1: Web Search (Tavily) - ONLY if strictly necessary
        if classification == "WEB_QUERY":
            messages.append(ChatMessage(type="thought", content="🌐 Query requires live data. Searching Tavily...", timestamp=datetime.now().isoformat()))
            
            # Perform Search
            context = await perform_web_search(req.query)
            
            # Log results snippet
            preview = (context[:200] + '...') if len(context) > 200 else context
            messages.append(ChatMessage(type="thought", content=f"📄 Web Data:\n{preview}", timestamp=datetime.now().isoformat()))
            
            messages.append(ChatMessage(type="thought", content="✨ Synthesizing answer from web results...", timestamp=datetime.now().isoformat()))
            final = await generate_web_answer(req.query, context)

        # PATH 2: General LLM (Default for everything else)
        # Note: PRODUCT_QUERY logic has been removed as requested.
        else:
            messages.append(ChatMessage(type="thought", content="🤖 Handling as general conversation...", timestamp=datetime.now().isoformat()))
            final = await generate_general_answer(req.query)
        
        return ChatResponse(messages=messages, final_answer=final)

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        print(error_message)
        messages.append(ChatMessage(type="thought", content=error_message, timestamp=datetime.now().isoformat()))
        return ChatResponse(messages=messages, final_answer="Sorry, I encountered an error.")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Server running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)