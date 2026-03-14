import os
import pickle
import re
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EXTRA_DOCS_DIR = "data/extra_docs/"
VECTOR_STORE_PATH = "data/vector_store.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

def get_embedding(text: str) -> list:
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}", "Content-Type": "application/json"},
        json={"model": "jina-embeddings-v3", "input": [text]}
    )
    return response.json()["data"][0]["embedding"]

def load_extra_docs(directory: str) -> str:
    text = ""
    path = Path(directory)
    if not path.exists():
        return text
    for txt_file in sorted(path.glob("*.txt")):
        print(f"  📄 Loading: {txt_file.name}")
        content = txt_file.read_text(encoding="utf-8")
        text += f"\n\n{content}"
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
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

    print("📄 Loading docs...")
    all_text = load_extra_docs(EXTRA_DOCS_DIR)

    if not all_text.strip():
        print("❌ No text found. Exiting.")
        return

    all_text = clean_text(all_text)
    chunks = chunk_text(all_text)
    print(f"✅ Created {len(chunks)} chunks")

    print(f"\n⏳ Embedding {len(chunks)} chunks via Jina...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(chunks)} done...")

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    print(f"\n🎉 Done! {len(chunks)} chunks saved")

if __name__ == "__main__":
    main()
