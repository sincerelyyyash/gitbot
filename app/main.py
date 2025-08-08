import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api import webhook_api, dashboard_api, admin_api
from app.config import settings
from app.services import rag_service
from app.services.indexing_service import indexing_service
from app.middleware.rate_limiter import RateLimitMiddleware
from app.core.database import init_db, close_db
from app.core.cache_manager import cache_manager, CacheBackend
from app.core.async_utils import async_executor
from app.core.payload_validator import payload_validator, webhook_validator, payload_rate_limiter
import asyncio
from google.api_core.exceptions import PermissionDenied, ResourceExhausted, InvalidArgument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from typing import Set, Dict, Any
from datetime import datetime, timedelta
import gc

app = FastAPI(title="GitBot")

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Background task manager for cleanup and monitoring
class BackgroundTaskManager:
    """Manages background tasks for cleanup and monitoring."""
    
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._task_stats = {
            "total_tasks": 0,
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0
        }
    
    async def add_task(self, task: asyncio.Task, task_type: str = "unknown") -> None:
        """Add a background task with monitoring."""
        self._tasks.add(task)
        self._task_stats["total_tasks"] += 1
        self._task_stats["active_tasks"] += 1
        
        # Add callback to track completion
        def task_done_callback(fut):
            self._task_stats["active_tasks"] -= 1
            if fut.exception():
                self._task_stats["failed_tasks"] += 1
                logger.error(f"Background task failed: {fut.exception()}")
            else:
                self._task_stats["completed_tasks"] += 1
            self._tasks.discard(fut)
        
        task.add_done_callback(task_done_callback)
    
    async def start_periodic_cleanup(self) -> None:
        """Start periodic cleanup tasks."""
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        await self.add_task(cleanup_task, "periodic_cleanup")
    
    async def start_cache_cleanup(self) -> None:
        """Start cache cleanup tasks."""
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup())
        await self.add_task(cache_cleanup_task, "cache_cleanup")
    
    async def start_performance_monitoring(self) -> None:
        """Start performance monitoring tasks."""
        monitoring_task = asyncio.create_task(self._performance_monitoring())
        await self.add_task(monitoring_task, "performance_monitoring")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of resources."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old background tasks
                await self._cleanup_old_tasks()
                
                # Force garbage collection
                gc.collect()
                
                # Log memory usage
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _cache_cleanup(self) -> None:
        """Periodic cache cleanup."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up expired cache entries
                await cache_manager.clear()  # This will clean up expired entries
                
                # Log cache statistics
                cache_stats = cache_manager.get_stats()
                logger.debug(f"Cache stats: {cache_stats}")
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring(self) -> None:
        """Monitor performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Get performance metrics
                async_stats = async_executor.get_stats()
                cache_stats = cache_manager.get_stats()
                payload_stats = payload_validator.get_validation_stats()
                
                # Log performance metrics
                logger.info(f"Performance metrics - "
                          f"Async ops: {async_stats['thread_pool_operations'] + async_stats['process_pool_operations']}, "
                          f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}, "
                          f"Payload violations: {payload_stats.get('violation_rate', 0):.2%}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks."""
        current_time = datetime.utcnow()
        tasks_to_remove = set()
        
        for task in self._tasks:
            if task.done():
                tasks_to_remove.add(task)
        
        for task in tasks_to_remove:
            self._tasks.discard(task)
            if task.exception():
                logger.warning(f"Cleaned up failed task: {task.exception()}")
    
    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        logger.info("Shutting down background task manager...")
        self._shutdown_event.set()
        
        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Background task manager shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        return {
            "total_tasks": self._task_stats["total_tasks"],
            "active_tasks": self._task_stats["active_tasks"],
            "completed_tasks": self._task_stats["completed_tasks"],
            "failed_tasks": self._task_stats["failed_tasks"],
            "task_types": list(set(task.get_name() for task in self._tasks if hasattr(task, 'get_name')))
        }

# Initialize background task manager
background_task_manager = BackgroundTaskManager()

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
    rate_limit=settings.rate_limit,
    window=settings.rate_limit_window,
    burst_limit=settings.rate_limit_burst,
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
            
    except PermissionDenied:
        logger.error("❌ Google API key validation failed - permission denied")
        return False
    except ResourceExhausted:
        logger.error("❌ Google API key validation failed - quota exceeded")
        return False
    except InvalidArgument:
        logger.error("❌ Google API key validation failed - invalid argument")
        return False
    except GoogleGenerativeAIError as e:
        logger.error(f"❌ Google API key validation failed - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Google API key validation failed - unexpected error: {e}")
        return False

async def initialize_performance_components():
    """Initialize performance-related components."""
    try:
        # Initialize cache manager
        if settings.cache_backend == "redis" and settings.redis_url:
            logger.info(f"Initializing Redis cache with URL: {settings.redis_url}")
            # Note: Cache manager is already initialized globally, but we can configure it
        else:
            logger.info("Using in-memory cache")
        
        # Initialize async executor with configuration
        logger.info(f"Initializing async executor with {settings.async_max_workers} workers")
        
        # Initialize payload validator with configuration
        logger.info("Initializing payload validator")
        
        # Initialize payload rate limiter
        logger.info(f"Initializing payload rate limiter: {settings.max_large_payloads} large payloads per {settings.large_payload_time_window}s")
        
        logger.info("✅ Performance components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize performance components: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue startup even if database fails
    
    # Validate Google API configuration
    api_valid = await validate_google_api_configuration()
    
    if not api_valid:
        logger.warning(
            "⚠️ Google API configuration issues detected. "
            "The bot will still run but with limited functionality. "
            "Users will receive helpful error messages when RAG features are unavailable."
        )
    
    # Initialize performance components
    perf_valid = await initialize_performance_components()
    if not perf_valid:
        logger.warning("⚠️ Performance components initialization failed. Using fallback configurations.")
    
    # Start indexing service
    await indexing_service.start()
    
    # Start background tasks with proper management
    await background_task_manager.start_periodic_cleanup()
    await background_task_manager.start_cache_cleanup()
    await background_task_manager.start_performance_monitoring()
    
    logger.info("🚀 GitBot startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down GitBot...")
    
    # Shutdown background task manager
    await background_task_manager.shutdown()
    
    # Shutdown async executor
    await async_executor.shutdown()
    
    # Stop indexing service
    await indexing_service.stop()
    
    # Close database connections
    await close_db()
    
    logger.info("✅ GitBot shutdown complete")

# Include API routers
app.include_router(webhook_api.router)
app.include_router(dashboard_api.router)
app.include_router(admin_api.router)

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "GitBot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

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
                api_message = "Google API returned empty result"
        except Exception as e:
            api_status = "unhealthy"
            api_message = f"Google API error: {str(e)}"
        
        # Get system metrics
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get background task status
        task_stats = background_task_manager.get_stats()
        
        # Get performance metrics
        cache_stats = cache_manager.get_stats()
        async_stats = async_executor.get_stats()
        payload_stats = payload_validator.get_validation_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": {
                    "status": api_status,
                    "message": api_message
                },
                "database": "healthy",  # Assuming database is healthy if we got here
                "indexing_service": "running" if indexing_service._workers else "stopped",
                "background_tasks": task_stats
            },
            "performance": {
                "memory": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                    "percent": round(process.memory_percent(), 2)
                },
                "cache": cache_stats,
                "async_operations": async_stats,
                "payload_validation": payload_stats
            },
            "configuration": {
                "cache_backend": settings.cache_backend,
                "async_workers": settings.async_max_workers,
                "max_payload_size_mb": round(settings.max_payload_size / 1024 / 1024, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with performance monitoring."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Log performance impact
    logger.error(f"Request that caused exception: {request.method} {request.url}")
    
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat()
    } 