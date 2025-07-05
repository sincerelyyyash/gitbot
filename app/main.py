import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.webhook import router as webhook_router
from app.config import settings
from app.services.rag_service import cleanup_inactive_collections
from app.services.indexing_service import indexing_service
from app.middleware.rate_limiter import RateLimitMiddleware
import asyncio
from google.api_core.exceptions import PermissionDenied, ResourceExhausted, InvalidArgument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

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

async def validate_google_api_configuration():
    """Validate Google API configuration on startup."""
    try:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set in .env. RAG functionality will be limited.")
            return False
        
        # Test the API key with a simple embedding request
        logger.info("Validating Google API key configuration...")
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.gemini_api_key,
            model="models/embedding-001"
        )
        
        # Try a minimal embedding request
        test_result = await asyncio.create_task(
            asyncio.to_thread(embeddings.embed_query, "test")
        )
        
        if test_result:
            logger.info("✅ Google API key validation successful")
            return True
        else:
            logger.error("❌ Google API key validation failed - no result returned")
            return False
            
    except (GoogleGenerativeAIError, PermissionDenied, ResourceExhausted, InvalidArgument) as e:
        error_str = str(e).lower()
        
        if "ip address restriction" in error_str or "api_key_ip_address_blocked" in error_str:
            logger.error(
                "❌ Google API key has IP address restrictions. "
                "Please either:\n"
                "  • Remove IP restrictions in Google Cloud Console, or\n"
                "  • Add your current server IP to the allowed list\n"
                f"  • Current error: {str(e)}"
            )
        elif "quota exceeded" in error_str:
            logger.warning(
                "⚠️ Google API quota exceeded. "
                "The application will handle this gracefully, but consider increasing your quota."
            )
        elif "invalid" in error_str and "api" in error_str:
            logger.error(
                "❌ Invalid Google API key. "
                "Please check your GEMINI_API_KEY configuration."
            )
        else:
            logger.error(f"❌ Google API configuration error: {str(e)}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during Google API validation: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    # Validate Google API configuration
    api_valid = await validate_google_api_configuration()
    
    if not api_valid:
        logger.warning(
            "⚠️ Google API configuration issues detected. "
            "The bot will still run but with limited functionality. "
            "Users will receive helpful error messages when RAG features are unavailable."
        )
    
    # Start indexing service
    await indexing_service.start()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_inactive_collections())
    background_tasks.add(cleanup_task)
    cleanup_task.add_done_callback(background_tasks.discard)
    
    logger.info("Started background tasks and indexing service")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop indexing service
    await indexing_service.stop()
    
    # Cancel all background tasks
    for task in background_tasks:
        task.cancel()
    
    # Wait for tasks to complete
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    logger.info("Shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint with comprehensive status information."""
    try:
        # Quick API validation (with timeout)
        api_status = "unknown"
        api_message = ""
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.gemini_api_key,
                model="models/embedding-001"
            )
            # Very short test with timeout
            test_result = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(embeddings.embed_query, ".")),
                timeout=5.0  # 5 second timeout
            )
            if test_result:
                api_status = "healthy"
                api_message = "Google API is accessible"
            else:
                api_status = "degraded"
                api_message = "Google API returned no result"
                
        except asyncio.TimeoutError:
            api_status = "timeout"
            api_message = "Google API request timed out"
        except Exception as e:
            error_str = str(e).lower()
            if "ip address restriction" in error_str or "api_key_ip_address_blocked" in error_str:
                api_status = "ip_restricted"
                api_message = "IP address restrictions on API key"
            elif "quota exceeded" in error_str:
                api_status = "quota_exceeded"
                api_message = "API quota exceeded"
            elif "invalid" in error_str:
                api_status = "invalid_key"
                api_message = "Invalid API key"
            else:
                api_status = "error"
                api_message = f"API error: {str(e)[:100]}"
        
        overall_status = "healthy" if api_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "version": "1.0.0",
            "environment": settings.environment,
            "google_api": {
                "status": api_status,
                "message": api_message
            },
            "features": {
                "rag_available": api_status in ["healthy", "quota_exceeded"],
                "fallback_responses": True
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "version": "1.0.0",
            "environment": settings.environment,
            "google_api": {
                "status": "unknown",
                "message": f"Health check error: {str(e)}"
            },
            "features": {
                "rag_available": False,
                "fallback_responses": True
            }
        }

# Include routers
app.include_router(webhook_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050) 