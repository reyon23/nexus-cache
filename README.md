# NexusCache Gateway (v0.3.0)

An intelligent, low-latency reverse proxy gateway for Large Language Models (LLMs) built with **FastAPI**, **Redis**, **Sentence-Transformers**, and **Prometheus**.

NexusCache sits between client applications and upstream LLM APIs (`POST /v1/chat/completions`), eliminating redundant token expenses and reducing response latency from seconds to milliseconds.

## Key Features

- **Tier-1 Exact Match Cache:** Deterministic SHA256 prompt hashing stored in Redis for sub-5ms responses (`X-Cache: HIT-EXACT`).
- **Tier-2 Semantic Vector Cache:** Local 384-dimensional vector embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) and NumPy cosine similarity search (`>= 0.85` threshold) to catch paraphrased prompts (`X-Cache: HIT-SEMANTIC`).
- **SSE Streaming Support (`stream: true`):** Full Server-Sent Events (`text/event-stream`) token streaming for both upstream cache misses and cached hits, sharing a unified cache across streaming and standard requests.
- **Production Telemetry & Cost Analytics:**
  - `GET /metrics`: Prometheus exposition format counters, gauges, and latency histograms.
  - `GET /stats`: Real-time JSON analytics summarizing cache hit ratio, tokens saved, and estimated USD cost savings.
- **Automated Stale Index Pruning:** Automatically cleans up expired Redis TTL entries during Tier-2 similarity scans.

## Architecture Pipeline

```text
Client Request ─► FastAPI Gateway ─► Tier-1 SHA256 Lookup ──(Hit)──► Return (< 5ms)
                                          │ (Miss)
                                          ▼
                                     Tier-2 Vector Search ──(>= 0.85)─► Return (< 25ms)
                                          │ (Miss)
                                          ▼
                                     Upstream LLM API ───► Store in Redis & Stream/Return
```

## Quick Start

1. **Start Redis via Docker Compose:**
   ```bash
   docker compose up -d
   ```

2. **Activate Virtual Environment & Install Dependencies:**
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the NexusCache Gateway:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion proxy (supports `"stream": true` and `false`) |
| `GET` | `/health` | Service status, Redis connection check, and active vector count |
| `GET` | `/stats` | Human-readable cache hit rate, tokens saved, and estimated USD savings |
| `GET` | `/metrics` | Prometheus-compatible metrics endpoint |