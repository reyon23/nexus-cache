import hashlib
import json
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import redis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NexusCache Proxy", version="0.1.0")

# Connect to local Redis container
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def generate_cache_key(payload: dict) -> str:
    """Generates a deterministic SHA256 hash from request messages."""
    messages = payload.get("messages", [])
    model = payload.get("model", "gpt-3.5-turbo")
    
    # Normalize JSON payload by sorting keys
    normalized_data = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return f"cache:{hashlib.sha256(normalized_data.encode('utf-8')).hexdigest()}"

@app.get("/health")
def health_check():
    """Verify proxy and Redis connectivity."""
    try:
        redis_status = redis_client.ping()
        return {"status": "online", "redis_connected": redis_status}
    except Exception as e:
        return {"status": "degraded", "redis_error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Intercepts OpenAI requests, checks Redis cache, or forwards upstream."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    cache_key = generate_cache_key(body)

    # 1. Check Redis for exact match (Tier-1 Cache)
    cached_response = redis_client.get(cache_key)
    if cached_response:
        response_data = json.loads(cached_response)
        return JSONResponse(
            content=response_data, 
            headers={"X-Cache": "HIT", "X-Cache-Engine": "Exact-SHA256"}
        )

    # 2. Cache Miss - Mock response if no valid OpenAI API key is provided
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        mock_response = {
            "id": "chatcmpl-nexuscache-mock",
            "object": "chat.completion",
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! This is a mock response from NexusCache proxy."
                },
                "finish_reason": "stop"
            }]
        }
        # Save to Redis for 1 hour (3600 seconds)
        redis_client.setex(cache_key, 3600, json.dumps(mock_response))
        return JSONResponse(
            content=mock_response, 
            headers={"X-Cache": "MISS", "X-Cache-Engine": "Exact-SHA256"}
        )

    # 3. Cache Miss - Forward request to Upstream OpenAI API
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            upstream_res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=30.0
            )
            response_json = upstream_res.json()
            
            # Store successful response in Redis
            if upstream_res.status_code == 200:
                redis_client.setex(cache_key, 3600, json.dumps(response_json))

            return JSONResponse(
                content=response_json, 
                status_code=upstream_res.status_code, 
                headers={"X-Cache": "MISS", "X-Cache-Engine": "Exact-SHA256"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upstream request failed: {str(e)}")