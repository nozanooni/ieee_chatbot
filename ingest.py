"""
ingest.py — Intelligent PDF extraction + RAG pipeline
- Uses Groq vision to extract text from ANY Arabic PDF (even designed booklets)
- Falls back to pdfplumber for simple text PDFs
- Embeds all chunks via Jina AI
- Saves to vector_store.pkl
"""

import os, pickle, re, requests, json, base64, time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH       = "data/ieee_kau.pdf"
EXTRA_DOCS_DIR = "data/extra_docs/"
VECTOR_STORE   = "data/vector_store.pkl"
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 100

GROQ_KEY = os.environ.get("GROQ_API_KEY")
JINA_KEY = os.environ.get("JINA_API_KEY")

# ── Jina embedding ────────────────────────────────────────────────────────────
def embed(text: str) -> list:
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_KEY}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text]}
    )
    return r.json()["data"][0]["embedding"]

# ── PDF → images via pdf2image ────────────────────────────────────────────────
def pdf_to_images(pdf_path: str) -> list:
    """Convert PDF pages to base64 PNG images."""
    try:
        from pdf2image import convert_from_path
        print("  Converting PDF pages to images...")
        pages = convert_from_path(pdf_path, dpi=150)
        images = []
        for page in pages:
            import io
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images.append(b64)
        print(f"  ✅ Converted {len(images)} pages to images")
        return images
    except ImportError:
        print("  ⚠️  pdf2image not installed. Run: uv pip install pdf2image")
        return []
    except Exception as e:
        print(f"  ⚠️  PDF to image failed: {e}")
        return []

# ── Groq vision extraction ────────────────────────────────────────────────────
def extract_page_with_vision(b64_image: str, page_num: int) -> str:
    """Send a page image to Groq vision model to extract Arabic text."""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                        {
                            "type": "text",
                            "text": """Extract ALL Arabic and English text from this image exactly as written.
Include all headings, body text, names, titles, labels, and any other text visible.
Output only the extracted text, nothing else. Preserve the logical reading order."""
                        }
                    ]
                }],
                "max_tokens": 2000,
                "temperature": 0
            },
            timeout=30
        )
        result = r.json()
        if "choices" in result:
            text = result["choices"][0]["message"]["content"].strip()
            print(f"  ✅ Page {page_num}: extracted {len(text)} chars")
            return text
        else:
            print(f"  ⚠️  Page {page_num}: {result.get('error', {}).get('message', 'unknown error')}")
            return ""
    except Exception as e:
        print(f"  ⚠️  Page {page_num} vision failed: {e}")
        return ""

# ── Simple pdfplumber fallback ────────────────────────────────────────────────
def extract_with_pdfplumber(pdf_path: str) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=5, y_tolerance=5, use_text_flow=True)
                if words:
                    text += " ".join(w["text"] for w in words) + "\n"
        return text
    except Exception as e:
        print(f"  pdfplumber failed: {e}")
        return ""

# ── Smart PDF extraction ──────────────────────────────────────────────────────
def extract_pdf(pdf_path: str) -> str:
    print("📄 Extracting PDF with AI vision...")

    # Step 1: Try vision extraction
    images = pdf_to_images(pdf_path)
    if images:
        all_text = ""
        for i, img in enumerate(images):
            text = extract_page_with_vision(img, i + 1)
            if text:
                all_text += f"\n\n{text}"
            time.sleep(0.5)  # rate limit buffer

        if len(all_text.strip()) > 200:
            print(f"✅ Vision extraction: {len(all_text)} chars from {len(images)} pages")
            return all_text

    # Step 2: Fallback to pdfplumber
    print("  Falling back to pdfplumber...")
    text = extract_with_pdfplumber(pdf_path)
    if text:
        print(f"✅ pdfplumber extraction: {len(text)} chars")
    return text

# ── Load extra docs ───────────────────────────────────────────────────────────
def load_extra_docs(directory: str) -> str:
    text = ""
    path = Path(directory)
    if not path.exists():
        return text
    for txt_file in sorted(path.glob("*.txt")):
        print(f"  📄 Loading: {txt_file.name}")
        text += f"\n\n{txt_file.read_text(encoding='utf-8')}"
    return text

# ── Clean text ────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r'[\u0640]+', '', text)    # remove tatweel
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ── Chunk text ────────────────────────────────────────────────────────────────
def chunk(text: str) -> list:
    sentences = re.split(r'(?<=[.،؟!\n])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= CHUNK_SIZE:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = overlap + " " + s
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 40]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(EXTRA_DOCS_DIR, exist_ok=True)
    all_text = ""

    # 1. PDF extraction (AI vision)
    if os.path.exists(PDF_PATH):
        pdf_text = extract_pdf(PDF_PATH)
        if pdf_text.strip():
            all_text += pdf_text
        else:
            print("⚠️  PDF extraction returned no text")
    else:
        print(f"⚠️  No PDF found at {PDF_PATH}")

    # 2. Extra docs (manual additions, tweets, etc.)
    extra = load_extra_docs(EXTRA_DOCS_DIR)
    if extra:
        all_text += extra
        print(f"✅ Extra docs loaded")

    if not all_text.strip():
        print("❌ No text found at all. Exiting.")
        return

    # 3. Clean + chunk
    all_text = clean(all_text)
    chunks = chunk(all_text)
    print(f"\n✅ Created {len(chunks)} chunks")
    print(f"📋 Sample chunk:\n{chunks[0][:200]}\n")

    # 4. Embed all chunks via Jina
    print(f"⏳ Embedding {len(chunks)} chunks via Jina AI...")
    embeddings = []
    for i, c in enumerate(chunks):
        emb = embed(c)
        embeddings.append(emb)
        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(chunks)} done...")

    # 5. Save
    with open(VECTOR_STORE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    print(f"\n🎉 Done! {len(chunks)} chunks embedded and saved to {VECTOR_STORE}")
    print(f"   PDF pages processed: {'yes (AI vision)' if os.path.exists(PDF_PATH) else 'no PDF found'}")
    print(f"   Extra docs: {len(list(Path(EXTRA_DOCS_DIR).glob('*.txt')))} files")
    print(f"\nNext: kill $(lsof -t -i:8000) && .venv/bin/uvicorn main:app")

if __name__ == "__main__":
    main()