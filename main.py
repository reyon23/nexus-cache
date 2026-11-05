import hashlib
import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import redis
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="NexusCache Gateway - Sprint 2", version="0.2.0")

# Redis Connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Load lightweight local embedding model (384-dimensional vectors)
print("Loading vector embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Vector engine initialized successfully.")

SIMILARITY_THRESHOLD = 0.85  # 85% similarity trigger for semantic cache hit

def generate_cache_key(payload: dict) -> str:
    """Tier-1: Deterministic SHA256 hash for exact string matching."""
    messages = payload.get("messages", [])
    model = payload.get("model", "gpt-3.5-turbo")
    normalized_data = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return f"cache:{hashlib.sha256(normalized_data.encode('utf-8')).hexdigest()}"

def extract_user_prompt(payload: dict) -> str:
    """Extracts the last user message text from the request payload."""
    messages = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""

def compute_cosine_similarity(vec1: list, vec2: list) -> float:
    """Computes cosine similarity between two 1D vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

@app.get("/health")
def health_check():
    try:
        redis_status = redis_client.ping()
        vector_count = redis_client.scard("semantic_keys")
        return {
            "status": "online",
            "redis_connected": redis_status,
            "vector_cache_entries": vector_count,
            "semantic_engine": "sentence-transformers/all-MiniLM-L6-v2"
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_prompt = extract_user_prompt(body)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No valid user prompt found in messages")

    cache_key = generate_cache_key(body)

    # -------------------------------------------------------------
    # TIER-1: Exact Match SHA256 Cache Check
    # -------------------------------------------------------------
    cached_response = redis_client.get(cache_key)
    if cached_response:
        response_data = json.loads(cached_response)
        return JSONResponse(
            content=response_data,
            headers={
                "X-Cache": "HIT-EXACT",
                "X-Cache-Engine": "SHA256-Exact",
                "X-Similarity-Score": "1.0000"
            }
        )

    # -------------------------------------------------------------
    # TIER-2: Semantic Vector Cache Check
    # -------------------------------------------------------------
    query_vector = embedding_model.encode(user_prompt).tolist()
    stored_semantic_keys = redis_client.smembers("semantic_keys")

    best_match_key = None
    highest_similarity = 0.0

    for sem_key in stored_semantic_keys:
        raw_data = redis_client.get(f"sem_data:{sem_key}")
        if raw_data:
            entry = json.loads(raw_data)
            stored_vector = entry.get("vector", [])
            similarity = compute_cosine_similarity(query_vector, stored_vector)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_key = sem_key

    # Check if best similarity crosses our 85% threshold
    if highest_similarity >= SIMILARITY_THRESHOLD and best_match_key:
        matched_data = json.loads(redis_client.get(f"sem_data:{best_match_key}"))
        return JSONResponse(
            content=matched_data["response"],
            headers={
                "X-Cache": "HIT-SEMANTIC",
                "X-Cache-Engine": "Cosine-Vector-Search",
                "X-Similarity-Score": f"{highest_similarity:.4f}",
                "X-Matched-Prompt": matched_data["prompt"]
            }
        )

    # -------------------------------------------------------------
    # CACHE MISS: Upstream Generation & Storage
    # -------------------------------------------------------------
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        mock_response = {
            "id": f"chatcmpl-nexuscache-{hashlib.md5(user_prompt.encode()).hexdigest()[:8]}",
            "object": "chat.completion",
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Response for prompt: '{user_prompt}' generated via NexusCache Proxy."
                },
                "finish_reason": "stop"
            }]
        }
        res_payload = mock_response
        status_code = 200
    else:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            upstream_res = await client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=30.0)
            res_payload = upstream_res.json()
            status_code = upstream_res.status_code

    if status_code == 200:
        # Save to Tier-1 (Exact Match)
        redis_client.setex(cache_key, 3600, json.dumps(res_payload))

        # Save to Tier-2 (Semantic Vector Index)
        sem_id = hashlib.sha256(user_prompt.encode()).hexdigest()[:16]
        semantic_entry = {
            "prompt": user_prompt,
            "vector": query_vector,
            "response": res_payload
        }
        redis_client.setex(f"sem_data:{sem_id}", 3600, json.dumps(semantic_entry))
        redis_client.sadd("semantic_keys", sem_id)

    return JSONResponse(
        content=res_payload,
        status_code=status_code,
        headers={
            "X-Cache": "MISS",
            "X-Cache-Engine": "Upstream-Model",
            "X-Similarity-Score": "0.0000"
        }
    )