from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import numpy as np
from typing import Optional
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

app = FastAPI(title="IEEE KAU Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

embedder = SentenceTransformer("intfloat/multilingual-e5-small")

VECTOR_STORE_PATH = "data/vector_store.pkl"

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return [], []
    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]

chunks, embeddings = load_vector_store()

def get_embedding(text: str):
    return embedder.encode(
        f"query: {text}",
        normalize_embeddings=True
    )


SYSTEM_PROMPT = """أنتِ مساعدة ذكية لنادي IEEE الطلابي في جامعة الملك عبدالعزيز (قسم الطالبات - كلية الحاسبات وتقنية المعلومات).
مهمتكِ هي الإجابة على أسئلة الطالبات حول النادي بشكل ودود ومفيد.

التعليمات:
- أجيبي دائماً باللغة العربية
- استخدمي المعلومات المقدمة لكِ فقط للإجابة
- إذا لم تعرفي الإجابة، قولي "عذراً، لا تتوفر لديّ هذه المعلومات حالياً، يمكنكِ التواصل معنا على kau.ieee.sb@gmail.com"
- كوني ودودة ومشجعة للطالبات على الانضمام للنادي
- لا تخترعي معلومات غير موجودة في السياق"""

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    response: str
    sources_used: int

def retrieve_context(query: str, top_k: int = 4) -> str:
    if not chunks:
        return ""
    query_embedding = get_embedding(query)

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
    context = retrieve_context(req.message)
    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nالمعلومات المتاحة:\n{context}" if context else SYSTEM_PROMPT
        }
    ]
    for msg in req.conversation_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": req.message})
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    response_text = completion.choices[0].message.content
    return ChatResponse(
        response=response_text,
        sources_used=len(context.split("---")) if context else 0
    )

@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_store": len(chunks)}