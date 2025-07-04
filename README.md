# gitbot

## Overview

gitbot is an AI-powered GitHub Assistant that automates Q&A for issues and comments using FastAPI, LangChain, ChromaDB, and Gemini LLM. It is designed for modularity, production-readiness, and easy extensibility.

## Features

- **GitHub Webhook Listener:** Receives GitHub webhook events (issue comments, issues) via a FastAPI backend.
- **Webhook Signature Verification:** Ensures all incoming webhooks are authentic using the configured secret.
- **GitHub App Authentication:** Authenticates as a GitHub App using JWT and installation tokens.
- **Retrieval-Augmented Generation (RAG):** Uses LangChain + ChromaDB + Gemini LLM to answer questions from issues/comments.
- **Per-Repository Knowledge Base:** Maintains a separate knowledge base for each repository (in-memory, demo only).
- **Modular, Production-Ready Structure:** Clean separation of API, models, services, and core logic.
- **Dockerized:** Easy to run in any environment.
- **Logging & Error Handling:** Logs key actions and errors for easier debugging.

## Project Structure

```
app/
  ├── __init__.py
  ├── main.py                # FastAPI app, includes routers
  ├── api/
  │   ├── __init__.py
  │   └── webhook.py         # Webhook endpoint & event routing
  ├── core/
  │   ├── __init__.py
  │   ├── github_utils.py    # GitHub API/auth helpers
  │   └── rag_system.py      # RAG system logic
  ├── models/
  │   ├── __init__.py
  │   └── github.py          # Pydantic models for GitHub payloads
  ├── services/
  │   ├── __init__.py
  │   └── rag_service.py     # Business logic for RAG & GitHub
  └── config.py              # Environment/config management

requirements.txt
Dockerfile
.dockerignore
.env.example
README.md
tests/
  └── test_webhook.py        # Example test for webhook endpoint
```

## Setup & Usage

### 1. Clone & Configure

```sh
git clone <repo-url>
cd gitbot
cp .env.example .env  # Fill in your GitHub App and Gemini credentials
```

### 2. Build & Run with Docker

```sh
docker build -t gitbot-backend .
docker run -p 8000:8000 --env-file .env gitbot-backend
```

### 3. Local Development

- Install dependencies: `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload`

### 4. Testing

```sh
pytest tests/
```

## Environment Variables

See `.env.example` for all required variables:
- `GITHUB_APP_ID`
- `GITHUB_PRIVATE_KEY`
- `GITHUB_WEBHOOK_SECRET`
- `GEMINI_API_KEY`
- `CHROMA_PERSIST_DIR`

## Limitations / Not Yet Supported

- No dynamic repo content ingestion (only current issue/comment is used as context)
- No scheduled or automatic re-indexing
- No advanced issue triage (labels, summaries, duplicates)
- No pull request review features
- No persistent vector database (in-memory ChromaDB for demo)
- No multi-channel support (GitHub webhooks only)
- No web UI or admin interface
- No tool usage or external integrations

## Contributing

PRs and issues welcome! See the code structure above for guidance on where to add new features or logic.


