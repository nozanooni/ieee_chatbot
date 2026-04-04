from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.genai as genai
import os, numpy as np, pickle, requests, json, re
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

app = FastAPI(title="IEEE KAU Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

VECTOR_STORE_PATH = "data/vector_store.pkl"
TWEETS_JSON_PATH  = "data/tweets.json"

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return [], []
    with open(VECTOR_STORE_PATH, "rb") as f:
        d = pickle.load(f)
    return d["chunks"], d["embeddings"]

chunks, embeddings = load_vector_store()

def load_tweets():
    if not os.path.exists(TWEETS_JSON_PATH):
        return []
    with open(TWEETS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_embed_cache = {}
def get_embedding(text: str) -> list:
    key = text.strip().lower()[:200]
    if key in _embed_cache:
        return _embed_cache[key]
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
                 "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text]}
    )
    result = r.json()["data"][0]["embedding"]
    _embed_cache[key] = result
    return result

SYSTEM_PROMPT_EN = """You are a concise assistant for the IEEE KAU student club.
CRITICAL: English only. Never use Arabic.

Rules:
- 2-4 sentences max for simple questions
- Use bullet points ONLY when listing 3+ items: - Item: description
- Bold key terms: **term**
- Answer ONLY what was asked. Nothing extra.
- NEVER mention social media, forms, email, or contact info unless the user explicitly asked
- NEVER say "follow us", "fill out the form", "contact us", "for more information" unless asked
- If asked about upcoming events: briefly mention the types of events the club holds (from context), then say the latest announcements are on the club's X account @ieee_kau_sb. Stop there.
- Never invent details not in the context
- Club is open to ALL female KAU students"""

SYSTEM_PROMPT_AR = """أنتِ مساعدة مختصرة لنادي IEEE في جامعة الملك عبدالعزيز.
اللهجة السعودية البسيطة فقط.

القواعد:
- 2-4 جمل كحد أقصى
- نقاط فقط لـ 3 عناصر أو أكثر: - العنصر: الشرح
- **عناوين** للمصطلحات المهمة
- أجيبي فقط على ما سُئل. لا شيء إضافي.
- لا تذكري وسائل التواصل أو النماذج أو البريد إلا إذا سُئلتِ صراحةً
- لا تقولي أبداً "تابعي حساباتنا" أو "سجّلي" أو "تواصلي معنا" إلا إذا سُئلتِ
- إذا سُئلتِ عن الفعاليات القادمة: اذكري بإيجاز أنواع الفعاليات، ثم قولي إن آخر الإعلانات على حساب X الرسمي @ieee_kau_sb. اكتفي بذلك.
- لا تخترعي تفاصيل
- النادي مفتوح لجميع طالبات جامعة الملك عبدالعزيز"""

GREETINGS = ["هلا","اهلا","مرحبا","السلام","هاي","hi","hello","hey","ok","okay","مساء","صباح"]

def is_greeting(t): return any(g in t.lower() for g in GREETINGS) and len(t.strip()) < 20

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

def retrieve_context(query, top_k=3):
    if not chunks or is_greeting(query):
        return ""
    qe = np.array(get_embedding(query))
    ea = np.array(embeddings)
    sims = np.dot(ea, qe) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(qe) + 1e-10)
    idxs = np.argsort(sims)[-top_k:][::-1]
    parts = [chunks[i][:300] for i in idxs if sims[i] > 0.2]
    return "\n\n---\n\n".join(parts)

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    response: str
    sources_used: int
    events: Optional[list] = []
    buttons: Optional[list] = []

EVENT_KW = [
    'event','events','upcoming','activities','workshop','competition',
    'lecture','seminar','conference','whats on',
    'فعالية','فعاليات','قادم','نشاط','ورشة','مسابقة','محاضرة','اخبار'
]
CONTACT_KW = [
    'contact','social','instagram','whatsapp','telegram','linkedin',
    'join','how to join','sign up','membership','reach','email',
    'تواصل','انضم','انستقرام','واتساب','تيليجرام','لينكد','بريد','ايميل',
    'كيف انضم','وسائل التواصل'
]
CUTOFF_PHRASES = [
    'to stay updated','follow us','follow the club','social media accounts',
    'you can also join','fill out','join form','for more information',
    'contact the club','contact us at','you can contact','you can reach',
    'more information','learn more','get in touch',
    'لمزيد','تواصلي','تابعي','سجّلي','انضمي','وسائل التواصل','للتواصل','للانضمام',
    'للمزيد من المعلومات'
]

@app.get("/app")
def serve_chat():
    return FileResponse("chat.html")

@app.get("/landing")
def serve_index():
    return FileResponse("index.html")

@app.get("/")
def root():
    return {"status": "IEEE KAU Chatbot running", "chunks_loaded": len(chunks)}

@app.get("/events")
def events_endpoint():
    return {"events": get_relevant_events()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    msg_lower  = req.message.lower()
    is_event   = any(k in msg_lower for k in EVENT_KW) and not any(k in msg_lower for k in CONTACT_KW)
    is_contact = any(k in msg_lower for k in CONTACT_KW)

    system  = SYSTEM_PROMPT_AR if req.language == "ar" else SYSTEM_PROMPT_EN
    context = retrieve_context(req.message)
    events  = []

    if is_event:
        events = get_relevant_events()

    system_msg = f"{system}\n\nContext:\n{context}" if context else system
    user_msg   = req.message

    # Build conversation string including history
    conv = system_msg + "\n\n"
    for m in req.conversation_history[-3:]:
        role = "User" if m["role"] == "user" else "Assistant"
        conv += f"{role}: {m['content']}\n"
    conv += f"User: {user_msg}"

    response = client.models.generate_content(
        model="gemini-1.5-flash-8b",
        contents=conv,
        config={"temperature": 0.1, "max_output_tokens": 400}
    )
    response_text = response.text

    # Strip all URLs and emails
    response_text = re.sub(r'https?://\S+', '', response_text)
    response_text = re.sub(r'[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}', '', response_text)

    # Cut filler if not a contact question
    if not is_contact:
        lines = response_text.split('\n')
        clean = []
        for line in lines:
            if any(p in line.lower() for p in CUTOFF_PHRASES):
                break
            clean.append(line)
        response_text = '\n'.join(clean)

    response_text = re.sub(r'[ \t]*:\s*\n(\s*\n)*', ':\n', response_text)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    response_text = response_text.strip()

    buttons = []
    if is_event:
        buttons = [
            {"type": "x", "url": "https://x.com/ieee_kau_sb", "label": "View on X"}
        ]
    elif is_contact:
        buttons = [
            {"type": "form",      "url": "https://docs.google.com/forms/d/e/1FAIpQLSc7qI9gJxhBd5TJZJPHBXQmNmRRocxjYDDE-ccPcoLNbKSoLw/viewform", "label": "Join Form"},
            {"type": "instagram", "url": "https://instagram.com/ieee_kau_sb",   "label": "Instagram"},
            {"type": "linkedin",  "url": "https://www.linkedin.com/company/ieeekau-sb-female/", "label": "LinkedIn"},
            {"type": "telegram",  "url": "https://t.me/FCITIEEEECLUB",          "label": "Telegram"},
            {"type": "whatsapp",  "url": "https://chat.whatsapp.com/K12WdGPDFNq8XZ0KzmNSEg", "label": "Research Community (WhatsApp only)"},
            {"type": "email",     "url": "mailto:kau.ieee.sb@gmail.com",         "label": "kau.ieee.sb@gmail.com"},
        ]

    return ChatResponse(
        response=response_text,
        sources_used=len(context.split("---")) if context else 0,
        events=events,
        buttons=buttons
    )

@app.get("/health")
def health():
    tweets = load_tweets()
    return {"status": "ok", "chunks": len(chunks),
            "tweets": len(tweets),
            "upcoming": sum(1 for t in tweets if t.get("is_upcoming"))}