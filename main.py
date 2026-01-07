import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# --- Tools ---
def bing_web_search(query: str):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_API_KEY"]}
    params = {"q": query, "count": 5}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return [p["url"] for p in response.json().get("webPages", {}).get("value", [])]

def fetch_webpage(url: str):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        return soup.get_text(separator="\n")
    except: return ""

# --- Pipeline Components ---
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ.get("GROQ_API_KEY"))
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def run_pipeline(topic: str):
    # 1. Planner
    planner_prompt = ChatPromptTemplate.from_template("Create a research plan for the topic: {topic}")
    plan = (planner_prompt | llm | StrOutputParser()).invoke({"topic": topic})
    print(f"--- PLAN ---\n{plan}\n")

    # 2. Question Generator
    q_gen_prompt = ChatPromptTemplate.from_template("Based on this plan, generate 3 specific search queries:\n{plan}")
    queries = (q_gen_prompt | llm | StrOutputParser()).invoke({"plan": plan}).split("\n")
    
    # 3. Web Search & 4. Content Fetch (RAG with ChromaDB)
    all_text = ""
    for q in queries[:3]:
        urls = bing_web_search(q.strip())
        for url in urls[:2]:
            all_text += fetch_webpage(url) + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([all_text])
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    relevant_docs = retriever.get_relevant_documents(topic)
    context = "\n".join([d.page_content for d in relevant_docs])

    # 5. Content Condenser
    condenser_prompt = ChatPromptTemplate.from_template("Condense the following information about {topic} into a concise summary:\n\n{context}")
    summary = (condenser_prompt | llm | StrOutputParser()).invoke({"topic": topic, "context": context})
    
    return summary

if __name__ == "__main__":
    topic = "Impact of AI on software engineering in 2025"
    result = run_pipeline(topic)
    print(f"--- FINAL SUMMARY ---\n{result}")
