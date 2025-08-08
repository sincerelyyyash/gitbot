import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator, Field, ConfigDict
import json
import logging
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = ConfigDict(
        extra='allow',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    # Environment settings
    environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)"
    )
    
    # GitHub App settings
    github_app_id: str = Field(
        ...,
        description="GitHub App ID from the GitHub App settings page",
        env="GITHUB_APP_ID"
    )
    github_private_key: str = Field(
        ...,
        description="GitHub App private key (contents of the .pem file)",
        env="GITHUB_PRIVATE_KEY"
    )
    github_webhook_secret: str = Field(
        ...,
        description="GitHub webhook secret for verifying webhook payloads",
        env="GITHUB_WEBHOOK_SECRET"
    )
    github_app_name: str = Field(
        default="synapticbot",
        description="The name of the GitHub App",
        env="GITHUB_APP_NAME"
    )
    
    # Gemini API settings
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key for LLM functionality",
        env="GEMINI_API_KEY"
    )
    
    # Storage settings
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base directory for all data storage"
    )
    chromadb_persist_dir: Optional[Path] = Field(
        default=None,
        description="ChromaDB persistence directory (default: {data_dir}/chroma)"
    )
    quota_persist_dir: Optional[Path] = Field(
        default=None,
        description="Quota persistence directory (default: {data_dir}/quota)"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # Admin settings
    admin_token: Optional[str] = Field(
        default=None,
        description="Admin token for accessing management endpoints",
        env="ADMIN_TOKEN"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
    )
    
    # Database settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/gitbot.db",
        description="Database URL for SQLAlchemy",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy echo for debugging",
        env="DATABASE_ECHO"
    )
    
    # Database connection pooling settings
    database_max_connections: int = Field(
        default=20,
        description="Maximum number of database connections in the pool",
        env="DATABASE_MAX_CONNECTIONS"
    )
    database_pool_timeout: int = Field(
        default=30,
        description="Database connection pool timeout in seconds",
        env="DATABASE_POOL_TIMEOUT"
    )
    database_pool_recycle: int = Field(
        default=3600,
        description="Database connection pool recycle time in seconds",
        env="DATABASE_POOL_RECYCLE"
    )
    database_pool_pre_ping: bool = Field(
        default=True,
        description="Enable database connection health checks",
        env="DATABASE_POOL_PRE_PING"
    )
    database_max_overflow: int = Field(
        default=10,
        description="Maximum overflow connections in the pool",
        env="DATABASE_MAX_OVERFLOW"
    )
    
    # Security Settings
    allowed_hosts: List[str] = [
        # "localhost",
        # "127.0.0.1",
        # "0.0.0.0",
        "*.github.com",  # For GitHub webhook calls
    ]
    
    # Rate Limiting
    rate_limit: int = 30
    rate_limit_window: int = 60
    rate_limit_burst: int = 5
    
    # Performance Settings
    # Cache settings
    cache_backend: str = Field(
        default="memory",
        description="Cache backend (memory, redis)",
        env="CACHE_BACKEND"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Default cache TTL in seconds",
        env="CACHE_TTL_SECONDS"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Maximum number of cache items",
        env="CACHE_MAX_SIZE"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for caching",
        env="REDIS_URL"
    )
    
    # Async operation settings
    async_max_workers: int = Field(
        default=10,
        description="Maximum number of async thread pool workers",
        env="ASYNC_MAX_WORKERS"
    )
    async_max_process_workers: int = Field(
        default=4,
        description="Maximum number of async process pool workers",
        env="ASYNC_MAX_PROCESS_WORKERS"
    )
    async_timeout_seconds: float = Field(
        default=30.0,
        description="Default timeout for async operations",
        env="ASYNC_TIMEOUT_SECONDS"
    )
    async_batch_size: int = Field(
        default=100,
        description="Default batch size for async operations",
        env="ASYNC_BATCH_SIZE"
    )
    
    # Payload validation settings
    max_payload_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request payload size in bytes",
        env="MAX_PAYLOAD_SIZE"
    )
    max_json_depth: int = Field(
        default=10,
        description="Maximum JSON nesting depth",
        env="MAX_JSON_DEPTH"
    )
    max_array_length: int = Field(
        default=10000,
        description="Maximum array length in JSON payloads",
        env="MAX_ARRAY_LENGTH"
    )
    max_string_length: int = Field(
        default=1000000,  # 1MB
        description="Maximum string length in JSON payloads",
        env="MAX_STRING_LENGTH"
    )
    max_object_keys: int = Field(
        default=1000,
        description="Maximum number of keys in JSON objects",
        env="MAX_OBJECT_KEYS"
    )
    payload_validation_timeout: float = Field(
        default=5.0,
        description="Timeout for payload validation in seconds",
        env="PAYLOAD_VALIDATION_TIMEOUT"
    )
    
    # Large payload rate limiting
    max_large_payloads: int = Field(
        default=10,
        description="Maximum number of large payloads per time window",
        env="MAX_LARGE_PAYLOADS"
    )
    large_payload_time_window: float = Field(
        default=60.0,
        description="Time window for large payload rate limiting in seconds",
        env="LARGE_PAYLOAD_TIME_WINDOW"
    )
    large_payload_threshold: int = Field(
        default=1024 * 1024,  # 1MB
        description="Size threshold for considering payload 'large'",
        env="LARGE_PAYLOAD_THRESHOLD"
    )
    
    @validator("chromadb_persist_dir", pre=True)
    def set_chromadb_dir(cls, v, values):
        if not v:
            return values["data_dir"] / "chroma"
        return v
    
    @validator("quota_persist_dir", pre=True)
    def set_quota_dir(cls, v, values):
        if not v:
            return values["data_dir"] / "quota"
        return v
    
    @validator("github_private_key")
    def validate_private_key(cls, v):
        """Validate and format the GitHub private key."""
        if not v:
            raise ValueError("GitHub private key is required")
        
        # Remove any whitespace and normalize newlines
        v = v.strip().replace('\r\n', '\n').replace('\r', '\n')
        
        # If key is already in PEM format, return as is
        if v.startswith("-----BEGIN RSA PRIVATE KEY-----"):
            return v
        
        # If key is base64 only, wrap in PEM format
        if not v.startswith("-----BEGIN"):
            v = f"-----BEGIN RSA PRIVATE KEY-----\n{v}\n-----END RSA PRIVATE KEY-----"
        
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        v = v.lower()
        if v not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {', '.join(valid_envs)}")
        return v
    
    def setup_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

# Initialize settings
try:
    settings = Settings()
except Exception as e:
    print("\nError loading settings:", str(e))
    raise 