import requests
import os
from dotenv import load_dotenv
load_dotenv()

response = requests.post(
    "https://api.jina.ai/v1/embeddings",
    headers={
        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "jina-embeddings-v3",
        "input": ["كيف أنضم للنادي؟"]
    }
)
print(response.status_code)
print(response.json())
