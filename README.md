# NexusCache Gateway

An intelligent reverse proxy gateway for Large Language Models (LLMs) built with FastAPI and Redis.

## Features
- **Tier-1 Exact Match Cache:** Instant sub-5ms lookups using deterministic SHA256 prompt hashing.
- **Tier-2 Semantic Vector Cache:** Vector embedding search using `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity matching (0.85 threshold) for paraphrased prompts.
- **Upstream Fallback:** Seamless fallback handling for OpenAI chat completion endpoints.

## Sprint 2 Quick Start
1. Start Redis: `docker compose up -d`
2. Start Gateway: `.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000`
3. Health Check: `curl.exe http://127.0.0.1:8000/health`