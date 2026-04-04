from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os, numpy as np, pickle, requests, json, re
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()

app = FastAPI(title="IEEE KAU Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VECTOR_STORE_PATH = "data/vector_store.pkl"
TWEETS_JSON_PATH  = "data/tweets.json"

# ── Vector store ──────────────────────────────────────────────────────────────
def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return [], []
    with open(VECTOR_STORE_PATH, "rb") as f:
        d = pickle.load(f)
    return d["chunks"], d["embeddings"]

chunks, embeddings = load_vector_store()

# ── Tweets ────────────────────────────────────────────────────────────────────
def load_tweets():
    if not os.path.exists(TWEETS_JSON_PATH):
        return []
    with open(TWEETS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Jina embedding ────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
                 "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text]}
    )
    return r.json()["data"][0]["embedding"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_AR = """أنتِ مساعدة لنادي IEEE في جامعة الملك عبدالعزيز.
تحدثي باللهجة السعودية البسيطة.

قواعد صارمة:
- الردود قصيرة ومنظمة (3-5 جمل) - لا تطولي أبداً
- لا تكرري أي معلومة في نفس الرد
- للقوائم استخدمي: **العنوان:** الشرح
- لا تذكري nitter.net — استخدمي x.com
- لا تخترعي أوقاتاً أو تفاصيل غير موجودة في السياق
- رابط الانضمام للنادي: https://docs.google.com/forms/d/e/1FAIpQLSc7qI9gJxhBd5TJZJPHBXQmNmRRocxjYDDE-ccPcoLNbKSoLw/viewform
- البريد الإلكتروني: kau.ieee.sb@gmail.com
- إذا سألوا عن واتساب: وضّحي أن مجموعة الواتساب هي خاصة بمجتمع البحث التابع للنادي وليست للانضمام العام
- إذا لم تعرفي: للأسف ما عندي هالمعلومة، تواصلي على kau.ieee.sb@gmail.com"""

SYSTEM_PROMPT_EN = """You are an assistant for the IEEE student club at King Abdulaziz University (Female Section).
CRITICAL: Respond ONLY in English. Never switch to Arabic even if the knowledge base is in Arabic.

Rules:
- SHORT organized answers (3-5 sentences max)
- For lists: **Title:** description
- Never repeat information in the same response
- Never invent details or times not explicitly in the context
- Never mention nitter.net — use x.com instead
- If asked about WhatsApp: clarify it is specifically for the Research Community sub-group, not general club membership
- Join form: https://docs.google.com/forms/d/e/1FAIpQLSc7qI9gJxhBd5TJZJPHBXQmNmRRocxjYDDE-ccPcoLNbKSoLw/viewform
- Email: kau.ieee.sb@gmail.com
- Unknown: I don't have that info, contact kau.ieee.sb@gmail.com"""

# ── Helpers ───────────────────────────────────────────────────────────────────
GREETINGS = ["هلا","اهلا","مرحبا","السلام","هاي","hi","hello","hey","ok","okay","مساء","صباح"]
EVENT_KW   = ['فعالية','فعاليات','حدث','أحداث','قادم','مسابقة','ورشة','محاضرة',
              'إعلان','منشور','تويتر','اكس','جديد','event','upcoming','announcement',
              'workshop','latest','news','twitter','post']

def is_greeting(t): return any(g in t.lower() for g in GREETINGS) and len(t.strip()) < 20
def is_event_query(t): return any(k in t.lower() for k in EVENT_KW)

def get_relevant_events(max_events=5):
    tweets = load_tweets()
    now = datetime.now(timezone.utc)
    upcoming, recent = [], []
    for tw in tweets:
        if tw.get("is_upcoming"):
            upcoming.append(tw)
        elif tw.get("date"):
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(tw["date_raw"])
                if (now - dt).days <= 14:
                    recent.append(tw)
            except: pass
    results = upcoming[:max_events]
    if len(results) < 3:
        results += recent[:max_events - len(results)]
    return results

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_context(query, top_k=5):
    if not chunks or is_greeting(query):
        return ""
    qe = np.array(get_embedding(query))
    ea = np.array(embeddings)
    sims = np.dot(ea, qe) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(qe) + 1e-10)
    idxs = np.argsort(sims)[-top_k:][::-1]
    parts = [chunks[i] for i in idxs if sims[i] > 0.2]
    return "\n\n---\n\n".join(parts)

# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    language: str = "ar"
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    response: str
    sources_used: int
    events: Optional[list] = []

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "IEEE KAU Chatbot ✅", "chunks_loaded": len(chunks)}

@app.get("/events")
def events_endpoint():
    return {"events": get_relevant_events()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    system = SYSTEM_PROMPT_AR if req.language == "ar" else SYSTEM_PROMPT_EN
    context = retrieve_context(req.message)
    events = []

    if is_event_query(req.message):
        events = get_relevant_events()
        if events:
            ev_ctx = "الفعاليات والإعلانات الأخيرة من حساب X الرسمي:\n"
            for e in events:
                date_str = e.get("date","")[:10] if e.get("date") else ""
                ev_ctx += f"- {e['title'][:120]} (تاريخ: {date_str}, رابط: {e.get('link','').replace('nitter.net','x.com')})\n"
            context = ev_ctx + "\n---\n\n" + context

    messages = [{"role": "system",
                 "content": f"{system}\n\nالمعلومات المتاحة:\n{context}" if context else system}]
    for m in req.conversation_history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return ChatResponse(
        response=completion.choices[0].message.content,
        sources_used=len(context.split("---")) if context else 0,
        events=events
    )

@app.get("/health")
def health():
    tweets = load_tweets()
    return {"status": "ok", "chunks": len(chunks),
            "tweets": len(tweets),
            "upcoming": sum(1 for t in tweets if t.get("is_upcoming"))}