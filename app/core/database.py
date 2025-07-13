"""
Database setup and configuration for GitBot.
"""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from app.config import settings

logger = logging.getLogger("database")

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    poolclass=StaticPool if settings.database_url.startswith("sqlite") else None,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    future=True
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create declarative base
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """
    Initialize database tables.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")

async def close_db():
    """
    Close database connections.
    """
    await engine.dispose()
    logger.info("Database connections closed") 