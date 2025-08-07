# gitbot

## Overview

gitbot is an AI-powered GitHub Assistant that automates Q&A, analyzes issues, and reviews pull requests using FastAPI, LangChain, ChromaDB, and the Gemini LLM. It is designed for modularity, production-readiness, and easy extensibility.

## Features

- **Advanced Issue Analysis:**
  - **Similarity Detection:** Identifies duplicate or related issues using a hybrid approach of semantic, keyword, and pattern-based similarity.
  - **Automated Triage:** Automatically comments on new issues with findings, suggests labels, and can identify invalid or incomplete issue reports.
- **Comprehensive Pull Request Review:**
  - **Automated Code Review:** Analyzes PRs for security vulnerabilities, code quality issues, complexity, and potential bugs.
  - **Actionable Feedback:** Posts PR reviews with detailed comments, assigns a review priority (`low` to `critical`), and adds labels like `security-risk` or `needs-refactor`.
- **Intelligent RAG System:**
  - **Automatic Content Indexing:** On installation, automatically indexes repository content including code, documentation, issues, and pull requests.
  - **Persistent Knowledge Base:** Uses a persistent ChromaDB vector store for each repository's knowledge base.
  - **Scheduled Re-indexing:** Periodically and automatically keeps the knowledge base up-to-date with repository changes.
- **Robust Infrastructure:**
  - **GitHub App Integration:** Authenticates as a GitHub App and handles a wide range of webhook events (installations, pushes, issues, PRs).
  - **Modular, Production-Ready Structure:** Clean separation of API, models, services, and core logic.
  - **Admin & Monitoring:** Includes admin endpoints for managing indexing and a circuit breaker for improved resilience.
  - **Dockerized:** Easy to run and deploy in any environment.

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
  │   ├── rate_limiter.py # Manages GitHub API rate limits
  │   ├── github_utils.py    # GitHub API/auth helpers
  │   ├── quota_manager.py   # Manages Gemini API quota
  │   └── rag_system.py      # RAG system logic
  ├── models/
  │   ├── __init__.py
  │   └── github.py          # Pydantic models for GitHub payloads
  ├── services/
  │   ├── __init__.py
  │   ├── indexing_service.py # Manages repository content indexing queue
  │   ├── issue_similarity_service.py # Logic for finding similar issues
  │   ├── pr_analysis_service.py # Logic for analyzing pull requests
  │   └── rag.py             # Business logic for RAG, issues, and PRs
  ├── middleware/
  │   └── rate_limiter.py      # Middleware for rate limiting
  └── config.py              # Environment/config management

requirements.txt
Dockerfile
.dockerignore
.env.example
README.md
tests/
  └── ... multiple test files ...
```

## Setup & Usage

### 1. Clone & Configure

```bash
git clone https://github.com/sincerelyyyash/gitbot.git
cd gitbot
cp .env.example .env  # Fill in your GitHub App and Gemini credentials
```

### 2. Build & Run with Docker

```bash
docker build -t gitbot-backend .
docker run -p 8050:8050 --env-file .env gitbot-backend
```

### 3. Local Development

- Install dependencies: `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload`

### 4. Testing

```bash
pytest tests/
```

## Environment Variables

See `.env.example` for all required variables:
- `GITHUB_APP_ID`
- `GITHUB_PRIVATE_KEY`
- `GITHUB_WEBHOOK_SECRET`
- `GEMINI_API_KEY`
- `CHROMA_PERSIST_DIR`
- `ADMIN_TOKEN`

## Limitations / Not Yet Supported

- No multi-channel support (GitHub webhooks only)
- No web UI or admin interface (admin tasks are API-based)
- Limited support for very large repositories (indexing can be time-consuming)
- Tool usage (e.g., running linters) is not yet implemented

## Contributing

PRs and issues welcome! See the code structure above for guidance on where to add new features or logic.


