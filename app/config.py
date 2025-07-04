import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator, Field, ConfigDict
import json
import logging

class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = ConfigDict(extra='allow')
    
    # Environment settings
    environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)"
    )
    
    # GitHub App settings
    github_app_id: str = Field(
        ...,
        description="GitHub App ID from the GitHub App settings page"
    )
    github_private_key: str = Field(
        ...,
        description="GitHub App private key (contents of the .pem file)"
    )
    github_webhook_secret: str = Field(
        ...,
        description="GitHub webhook secret for verifying webhook payloads"
    )
    
    # Gemini API settings
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key for LLM functionality"
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
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
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
        if not v.strip().startswith("-----BEGIN RSA PRIVATE KEY-----"):
            raise ValueError("Invalid private key format")
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
settings = Settings() 