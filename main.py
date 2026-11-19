import asyncio
import hashlib
import json
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import redis
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

app = FastAPI(title="NexusCache Gateway - Sprint 3", version="0.3.0")

# Redis Connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SIMILARITY_THRESHOLD = 0.85  # 85% cosine similarity trigger
CACHE_TTL_SECONDS = 3600     # 1-hour cache expiration
COST_PER_1K_TOKENS = 0.002   # Blended GPT-3.5-Turbo rate ($0.002 / 1K tokens)

# -------------------------------------------------------------
# Prometheus Telemetry Metrics
# -------------------------------------------------------------
REQUEST_COUNTER = Counter(
    "nexuscache_requests_total",
    "Total chat completion requests processed by NexusCache",
    ["cache_status", "stream_mode"]
)
REQUEST_LATENCY = Histogram(
    "nexuscache_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["cache_status"]
)
TOKENS_SAVED_COUNTER = Counter(
    "nexuscache_tokens_saved_total",
    "Estimated LLM tokens saved via cache hits"
)
VECTOR_INDEX_GAUGE = Gauge(
    "nexuscache_semantic_entries",
    "Number of active semantic vector entries in Redis"
)

# Load lightweight local embedding model (384-dimensional vectors)
print("Loading vector embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Vector engine initialized successfully.")


def generate_cache_key(payload: dict) -> str:
    """Tier-1: Deterministic SHA256 hash independent of stream flag."""
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


def estimate_tokens(prompt: str, response_text: str) -> int:
    """Estimates combined prompt + completion tokens (~1.3 tokens per word)."""
    word_count = len(prompt.split()) + len(response_text.split())
    return max(50, int(word_count * 1.3))


def record_telemetry(cache_status: str, is_stream: bool, duration_sec: float, tokens_saved: int = 0):
    """Updates both Prometheus metrics and persistent Redis analytics counters."""
    stream_label = "stream" if is_stream else "standard"
    REQUEST_COUNTER.labels(cache_status=cache_status, stream_mode=stream_label).inc()
    REQUEST_LATENCY.labels(cache_status=cache_status).observe(duration_sec)

    try:
        redis_client.incr("stats:total_requests")
        if cache_status == "HIT-EXACT":
            redis_client.incr("stats:hit_exact")
        elif cache_status == "HIT-SEMANTIC":
            redis_client.incr("stats:hit_semantic")
        else:
            redis_client.incr("stats:miss")

        if tokens_saved > 0:
            TOKENS_SAVED_COUNTER.inc(tokens_saved)
            redis_client.incrby("stats:tokens_saved", tokens_saved)

        VECTOR_INDEX_GAUGE.set(redis_client.scard("semantic_keys"))
    except Exception:
        pass


async def sse_stream_generator(content: str, model: str, completion_id: str):
    """Formats a completion string as OpenAI-compatible Server-Sent Events (SSE)."""
    words = content.split(" ")
    for idx, word in enumerate(words):
        token_piece = word if idx == 0 else f" {word}"
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token_piece},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.02)

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health_check():
    try:
        redis_status = redis_client.ping()
        vector_count = redis_client.scard("semantic_keys")
        VECTOR_INDEX_GAUGE.set(vector_count)
        return {
            "status": "online",
            "version": "0.3.0",
            "redis_connected": redis_status,
            "vector_cache_entries": vector_count,
            "semantic_engine": "sentence-transformers/all-MiniLM-L6-v2",
            "streaming_supported": True
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/metrics")
def prometheus_metrics():
    """Exposes metrics in Prometheus text exposition format."""
    try:
        VECTOR_INDEX_GAUGE.set(redis_client.scard("semantic_keys"))
    except Exception:
        pass
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
def cache_statistics():
    """Human-readable telemetry dashboard summarizing hit rate and cost savings."""
    total = int(redis_client.get("stats:total_requests") or 0)
    hit_exact = int(redis_client.get("stats:hit_exact") or 0)
    hit_semantic = int(redis_client.get("stats:hit_semantic") or 0)
    misses = int(redis_client.get("stats:miss") or 0)
    tokens_saved = int(redis_client.get("stats:tokens_saved") or 0)

    total_hits = hit_exact + hit_semantic
    hit_rate_pct = round((total_hits / total) * 100, 2) if total > 0 else 0.0
    estimated_usd_saved = round((tokens_saved / 1000.0) * COST_PER_1K_TOKENS, 6)

    return {
        "total_requests": total,
        "cache_hits": {
            "total": total_hits,
            "tier_1_exact": hit_exact,
            "tier_2_semantic": hit_semantic
        },
        "cache_misses": misses,
        "hit_rate_percent": hit_rate_pct,
        "estimated_tokens_saved": tokens_saved,
        "estimated_cost_saved_usd": estimated_usd_saved,
        "active_vector_entries": redis_client.scard("semantic_keys")
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start_time = time.perf_counter()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_prompt = extract_user_prompt(body)
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No valid user prompt found in messages")

    is_stream = bool(body.get("stream", False))
    model_name = body.get("model", "gpt-3.5-turbo")
    cache_key = generate_cache_key(body)

    # -------------------------------------------------------------
    # TIER-1: Exact Match SHA256 Cache Check
    # -------------------------------------------------------------
    cached_response = redis_client.get(cache_key)
    if cached_response:
        response_data = json.loads(cached_response)
        answer_text = response_data["choices"][0]["message"]["content"]
        saved = estimate_tokens(user_prompt, answer_text)
        elapsed = time.perf_counter() - start_time
        record_telemetry("HIT-EXACT", is_stream, elapsed, saved)

        headers = {
            "X-Cache": "HIT-EXACT",
            "X-Cache-Engine": "SHA256-Exact",
            "X-Similarity-Score": "1.0000"
        }
        if is_stream:
            return StreamingResponse(
                sse_stream_generator(answer_text, model_name, response_data.get("id", "chatcmpl-cache")),
                media_type="text/event-stream",
                headers=headers
            )
        return JSONResponse(content=response_data, headers=headers)

    # -------------------------------------------------------------
    # TIER-2: Semantic Vector Cache Check (with Stale Key Pruning)
    # -------------------------------------------------------------
    query_vector = embedding_model.encode(user_prompt).tolist()
    stored_semantic_keys = redis_client.smembers("semantic_keys")

    best_match_key = None
    highest_similarity = 0.0

    for sem_key in stored_semantic_keys:
        raw_data = redis_client.get(f"sem_data:{sem_key}")
        if not raw_data:
            # Prune expired TTL key from Redis set
            redis_client.srem("semantic_keys", sem_key)
            continue

        entry = json.loads(raw_data)
        stored_vector = entry.get("vector", [])
        similarity = compute_cosine_similarity(query_vector, stored_vector)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_key = sem_key

    if highest_similarity >= SIMILARITY_THRESHOLD and best_match_key:
        matched_data = json.loads(redis_client.get(f"sem_data:{best_match_key}"))
        response_data = matched_data["response"]
        answer_text = response_data["choices"][0]["message"]["content"]
        saved = estimate_tokens(user_prompt, answer_text)
        elapsed = time.perf_counter() - start_time
        record_telemetry("HIT-SEMANTIC", is_stream, elapsed, saved)

        headers = {
            "X-Cache": "HIT-SEMANTIC",
            "X-Cache-Engine": "Cosine-Vector-Search",
            "X-Similarity-Score": f"{highest_similarity:.4f}",
            "X-Matched-Prompt": matched_data["prompt"]
        }
        if is_stream:
            return StreamingResponse(
                sse_stream_generator(answer_text, model_name, response_data.get("id", "chatcmpl-sem")),
                media_type="text/event-stream",
                headers=headers
            )
        return JSONResponse(content=response_data, headers=headers)

    # -------------------------------------------------------------
    # CACHE MISS: Upstream Generation & Storage
    # -------------------------------------------------------------
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        completion_id = f"chatcmpl-nexuscache-{hashlib.md5(user_prompt.encode()).hexdigest()[:8]}"
        answer_text = f"Response for prompt: '{user_prompt}' generated via NexusCache Proxy."
        res_payload = {
            "id": completion_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text
                },
                "finish_reason": "stop"
            }]
        }
        status_code = 200
    else:
        upstream_body = dict(body)
        upstream_body["stream"] = False
        async with httpx.AsyncClient() as client:
            req_headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            upstream_res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=upstream_body,
                headers=req_headers,
                timeout=30.0
            )
            res_payload = upstream_res.json()
            status_code = upstream_res.status_code
            answer_text = res_payload.get("choices", [{}])[0].get("message", {}).get("content", "")

    if status_code == 200:
        # Save to Tier-1 (Exact Match)
        redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(res_payload))

        # Save to Tier-2 (Semantic Vector Index)
        sem_id = hashlib.sha256(user_prompt.encode()).hexdigest()[:16]
        semantic_entry = {
            "prompt": user_prompt,
            "vector": query_vector,
            "response": res_payload
        }
        redis_client.setex(f"sem_data:{sem_id}", CACHE_TTL_SECONDS, json.dumps(semantic_entry))
        redis_client.sadd("semantic_keys", sem_id)

    elapsed = time.perf_counter() - start_time
    record_telemetry("MISS", is_stream, elapsed, 0)

    miss_headers = {
        "X-Cache": "MISS",
        "X-Cache-Engine": "Upstream-Model",
        "X-Similarity-Score": "0.0000"
    }

    if is_stream and status_code == 200:
        return StreamingResponse(
            sse_stream_generator(answer_text, model_name, res_payload.get("id", "chatcmpl-miss")),
            media_type="text/event-stream",
            headers=miss_headers
        )

    return JSONResponse(content=res_payload, status_code=status_code, headers=miss_headers)