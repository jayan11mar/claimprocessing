# Docker Deployment — Claims Processing Assistant (Phase W8-8)

Containerisation artefacts for the complete Week 8 system. A single
`docker compose up --build` brings up the API + UI with volume-mounted
persistence and env-injected secrets, from a clean machine.

> **Status:** Optional phase (spec 5.2 / 10). Artefacts authored and
> **validated via `docker compose config`** (Docker Compose v5.1.4). Runtime
> `docker compose up` was deferred on the shared training VM — the Docker
> daemon cannot start in that restricted environment (kernel lacks
> netlink / bridge / iptables capabilities, even with `sudo` and
> `--iptables=false`). Run the commands below on any Docker-capable host to
> complete the runtime verification.

---

## Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Python 3.11-slim base, pinned `requirements.txt`, layer caching, `curl` for healthcheck, non-root `appuser`, **no baked secrets**. HF cache pinned to `/app/.hf_cache`. |
| `docker-compose.yml` | Two services — `api` (FastAPI/uvicorn :8000) and `ui` (Streamlit :8501); bind-mount `./data` (FAISS index + SQLite/HITL DB); named `hf_cache` volume (reranker model); `/health` healthcheck; secrets via `env_file` only. |
| `.dockerignore` | Excludes `.env`, `venv/`, caches, diagnostics, `*_SUMMARY.md`, plan files, `notebooks/`, `Screenshots/`, and `data/` artefacts (mounted at runtime, not baked). Keeps `.env.example` tracked. |

---

## Prerequisites

- Docker Engine + Compose **V2** (`docker compose`, space syntax).
- A populated `.env` (copy from `.env.example` and fill required keys):
  ```bash
  cp .env.example .env
  ```
- Local `data/` directory containing the FAISS index and SQLite DB
  (created by prior ingestion phases) — mounted into the container.

---

## Run (on a Docker-capable host)

```bash
# 1. Validate compose syntax + env resolution
docker compose config

# 2. Build and start both services
docker compose up --build -d

# 3. Health check (expect HTTP 200)
sleep 20 && curl -s localhost:8000/health

# 4. Stop the stack
docker compose down

# 5. Restart and re-check health
docker compose up -d && sleep 20 && curl -s localhost:8000/health

# 6. Prove persistence — these survive the down/up cycle
ls -la data/faiss_index data/claims.db
```

**Endpoints once up:**
- API  → http://localhost:8000  (health at `/health`)
- UI   → http://localhost:8501

---

## Acceptance criteria — mapping

| Criterion (spec) | How it's met |
|------------------|--------------|
| `docker compose up` yields working API + UI on a clean machine | `api` (:8000) + `ui` (:8501) services build from one `Dockerfile`; verified by `/health` 200 + UI reachable |
| No secrets in the image; `.env` injected at runtime only | Secrets sourced via `env_file: [.env]`; `.env` excluded by `.dockerignore`; Dockerfile bakes no keys |
| Persistent volumes retain vector index + HITL tasks across restarts | `./data:/app/data` bind-mount persists `faiss_index/` + `claims.db`; `hf_cache` named volume persists reranker model |

---

## Notes

- **HF cache override:** `Dockerfile` sets `HF_HOME` / `TRANSFORMERS_CACHE`
  to `/app/.hf_cache` (backed by the `hf_cache` volume). Keep the host-path
  cache lines in `.env.example` **commented** so `env_file` does not override
  the container ENV — otherwise the reranker model re-downloads every restart.
- **Disk:** `python:3.11-slim` + deps + the cross-encoder model total
  ≈ 1–1.5 GB. On constrained hosts, check `docker system df` before building
  and reclaim with `docker image prune -f` afterwards.
- **Compose command:** this environment ships Compose V2 — always use
  `docker compose` (space), not the legacy `docker-compose` (hyphen).
