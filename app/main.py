import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
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

# Add Trusted Host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=30,  # 30 requests per minute
    window=60,  # 1 minute window
    burst_limit=5,  # Allow 5 burst requests
    exclude_paths={"/health", "/api/webhook"}  # Don't rate limit health checks and GitHub webhooks
)

# Global exception handler for 404s
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"}
    )

# Global exception handler for 500s
@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.middleware("http")
async def handle_requests(request: Request, call_next):
    """Global middleware to handle all requests."""
    try:
        # Block access to sensitive files
        path = request.url.path.lower()
        if any(sensitive in path for sensitive in ['.env', '.git', '.htaccess', 'wp-', 'admin', 'favicon.ico']):
            return JSONResponse(
                status_code=403,
                content={"error": "Access forbidden"}
            )

        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    except Exception as e:
        logger.exception("Unexpected error in request handler")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/")
async def root():
    """Root endpoint that redirects to webhook path."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request",
            "message": "This endpoint only accepts webhooks at /api/webhook"
        }
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str):
    """Catch-all route handler for undefined routes."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request",
            "message": "This endpoint only accepts webhooks at /api/webhook"
        }
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

# Include routers
app.include_router(webhook_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050) 