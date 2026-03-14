from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import numpy as np
import pickle
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="IEEE KAU Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VECTOR_STORE_PATH = "data/vector_store.pkl"

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return [], []
    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]

chunks, embeddings = load_vector_store()

def get_embedding(text: str) -> list:
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text]}
    )
    return response.json()["data"][0]["embedding"]

GREETINGS = ["هلا","اهلا","مرحبا","السلام","هاي","hi","hello","hey","مساء","صباح","كيف حالك","ok","okay"]

def is_greeting(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(g in text_lower for g in GREETINGS) and len(text_lower) < 20

SYSTEM_PROMPT_AR = """أنتِ مساعدة لنادي IEEE في جامعة الملك عبدالعزيز.
تحدثي باللهجة السعودية البسيطة، اجعلي الردود قصيرة (2-3 جمل).
أجيبي فقط عن أسئلة متعلقة بنادي IEEE.
إذا كان السؤال تحية أو كلام عام، ردي بترحيب ودودة واسألي كيف تقدرين تساعدين.
إذا لم تعرفي الإجابة قولي: للأسف ما عندي هالمعلومة، تقدري تتواصلي معنا على kau.ieee.sb@gmail.com"""

SYSTEM_PROMPT_EN = """You are an assistant for the IEEE student club at King Abdulaziz University.
Keep answers short (2-3 sentences). Use clear simple English.
If the message is a greeting, respond warmly and ask how you can help.
Only answer questions about the IEEE club."""

class ChatRequest(BaseModel):
    message: str
    language: str = "ar"
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    response: str
    sources_used: int
    events: Optional[list] = []

def retrieve_context(query: str, top_k: int = 4) -> str:
    if not chunks or is_greeting(query):
        return ""
    query_embedding = np.array(get_embedding(query))
    embeddings_array = np.array(embeddings)
    similarities = np.dot(embeddings_array, query_embedding) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context_parts = [chunks[i] for i in top_indices if similarities[i] > 0.3]
    return "\n\n---\n\n".join(context_parts)

@app.get("/")
def root():
    return {"status": "IEEE KAU Chatbot is running ✅", "chunks_loaded": len(chunks)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    system_prompt = SYSTEM_PROMPT_AR if req.language == "ar" else SYSTEM_PROMPT_EN
    context = retrieve_context(req.message)
    messages = [{"role": "system", "content": f"{system_prompt}\n\nالمعلومات المتاحة:\n{context}" if context else system_prompt}]
    for msg in req.conversation_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": req.message})
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    response_text = completion.choices[0].message.content
    return ChatResponse(
        response=response_text,
        sources_used=len(context.split("---")) if context else 0,
        events=[]
    )

@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_store": len(chunks)}
