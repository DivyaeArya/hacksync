import os
import json
import asyncio
import requests
from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize LLM
llm = ChatGroq(
    temperature=0.1,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

def search_firecrawl(query: str):
    """The function from your snippet logic"""
    payload = {
        "query": query,
        "sources": ["web", "images", "news"],
        "limit": 5,
        "scrapeOptions": {"onlyMainContent": False}
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(FIRECRAWL_URL, json=payload, headers=headers)
    return response.json()

async def agent_pipeline(topic: str):
    """LangChain Pipeline that yields progress for the terminal"""
    
    # THOUGHT 1: Analyzing Topic
    yield f"data: {json.dumps({'type': 'thought', 'content': f'Analyzing topic: {topic}...'})}\n\n"
    await asyncio.sleep(1)

    # THOUGHT 2: Generating Questions
    yield f"data: {json.dumps({'type': 'thought', 'content': 'Invoking Groq LLM to generate 3 optimized search queries...'})}\n\n"
    
    prompt = ChatPromptTemplate.from_template(
        "You are a research agent. Generate exactly 3 distinct, high-quality search queries to find the latest news and images for the topic: {topic}. "
        "Output ONLY the queries, one per line, no numbering."
    )
    chain = prompt | llm | StrOutputParser()
    
    
    raw_queries = await chain.ainvoke({"topic": topic})
    queries = [q.strip() for q in raw_queries.split('\n') if q.strip()][:3]
    
    yield f"data: {json.dumps({'type': 'thought', 'content': f'Queries generated: {queries}'})}\n\n"
    await asyncio.sleep(1)

    # THOUGHT 3: Fetching Data
    all_results = []
    for i, q in enumerate(queries):
        yield f"data: {json.dumps({'type': 'thought', 'content': f'Searching Firecrawl for: {q}...'})}\n\n"
        # Run in thread to not block async loop
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, search_firecrawl, q)
        
        if res.get('success'):
            all_results.append(res.get('data', {}))
    
    # FINAL: Send Results
    yield f"data: {json.dumps({'type': 'result', 'content': all_results})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search-stream")
async def search_stream(topic: str):
    return StreamingResponse(agent_pipeline(topic), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)