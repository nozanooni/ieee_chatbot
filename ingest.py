import os
import pickle
import re
import requests
import numpy as np
from pathlib import Path
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-small")


load_dotenv()

PDF_PATH = "data/ieee_kau.pdf"
EXTRA_DOCS_DIR = "data/extra_docs/"
VECTOR_STORE_PATH = "data/vector_store.pkl"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def get_embedding(text: str):
    return embedder.encode(
        f"passage: {text}",
        normalize_embeddings=True
    )


def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[صفحة {page_num + 1}]\n{page_text}"
    return text

def load_extra_docs(directory: str) -> str:
    text = ""
    path = Path(directory)
    if not path.exists():
        return text
    for txt_file in path.glob("*.txt"):
        print(f"  📄 Loading: {txt_file.name}")
        text += f"\n\n[{txt_file.stem}]\n"
        text += txt_file.read_text(encoding="utf-8")
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\[صفحة \d+\]', '', text)
    return text.strip()

def chunk_text(text: str) -> list:
    sentences = re.split(r'(?<=[.،؟!\n])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
            current_chunk = overlap_text + " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 30]

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(EXTRA_DOCS_DIR, exist_ok=True)
    all_text = ""
    if os.path.exists(PDF_PATH):
        print("📄 Extracting text from PDF...")
        pdf_text = extract_pdf_text(PDF_PATH)
        all_text += pdf_text
        print(f"✅ Extracted {len(pdf_text)} characters from PDF")
    else:
        print(f"⚠️  PDF not found at {PDF_PATH}")
    extra_text = load_extra_docs(EXTRA_DOCS_DIR)
    if extra_text:
        all_text += extra_text
    else:
        print("ℹ️  No extra docs found (thats fine)")
    if not all_text.strip():
        print("❌ No text found. Exiting.")
        return
    all_text = clean_text(all_text)
    chunks = chunk_text(all_text)
    print(f"✅ Created {len(chunks)} chunks")
    print(f"⏳ Embedding {len(chunks)} chunks via Jina...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        if (i + 1) % 5 == 0:
            print(f"   {i + 1}/{len(chunks)} done...")
    print(f"✅ Created {len(embeddings)} embeddings")
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"\n🎉 Done! Vector store saved with {len(chunks)} chunks")
    print(f"Next step: uvicorn main:app --reload")

if __name__ == "__main__":
    main()
