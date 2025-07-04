import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.webhook import router as webhook_router
from app.config import settings
from app.services.rag_service import cleanup_inactive_collections
from app.middleware.rate_limiter import RateLimitMiddleware
import asyncio

app = FastAPI(title="GitBot")

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Store background tasks
background_tasks = set()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=30,  # 30 requests per minute
    window=60,  # 1 minute window
    burst_limit=5,  # Allow 5 burst requests
    exclude_paths={"/health"}  # Don't rate limit health checks
)

@app.on_event("startup")
async def startup_event():
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set in .env. RAG functionality will be limited.")
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_inactive_collections())
    background_tasks.add(cleanup_task)
    cleanup_task.add_done_callback(background_tasks.discard)
    
    logger.info("Started background tasks")

@app.on_event("shutdown")
async def shutdown_event():
    # Cancel all background tasks
    for task in background_tasks:
        task.cancel()
    
    # Wait for tasks to complete
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    logger.info("Shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.environment
    }

# Include routers
app.include_router(webhook_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050) 