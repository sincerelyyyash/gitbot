"""
Database Manager

Provides database management and session handling for GitBot including:
- Connection pooling and management
- Session lifecycle management
- Database health monitoring
- Performance metrics and optimization
- Migration support
"""

import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import asyncio
from .base import BaseCore, core_operation
from app.config import settings

# Create declarative base
Base = declarative_base()

class DatabaseManager(BaseCore):
    """
    Database manager providing connection pooling, session management, and monitoring.
    
    Features:
    - Connection pooling with health checks
    - Session lifecycle management
    - Database performance monitoring
    - Migration support
    - Connection retry logic
    """
    
    def __init__(self):
        super().__init__("database")
        
        # Database configuration
        self.database_url = settings.database_url
        self.database_echo = settings.database_echo
        self.max_connections = getattr(settings, 'database_max_connections', 20)
        self.pool_timeout = getattr(settings, 'database_pool_timeout', 30)
        self.pool_recycle = getattr(settings, 'database_pool_recycle', 3600)
        
        # Connection state
        self._engine = None
        self._session_factory = None
        self._connection_pool = None
        self._last_health_check = None
        self._health_check_interval = 300  # 5 minutes
        
        # Performance metrics
        self._connection_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "connection_errors": 0,
            "slow_queries": 0
        }
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database engine and session factory."""
        try:
            # Configure engine based on database type
            engine_kwargs = {
                "echo": self.database_echo,
                "future": True,
                "pool_size": min(self.max_connections, 10),
                "max_overflow": getattr(settings, 'database_max_overflow', 10),
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle,
                "pool_pre_ping": getattr(settings, 'database_pool_pre_ping', True),
                "pool_reset_on_return": "commit",  # Reset connections on return
            }
            
            # SQLite specific configuration
            if self.database_url.startswith("sqlite"):
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False}
                })
            
            # Create async engine
            self._engine = create_async_engine(
                self.database_url,
                **engine_kwargs
            )
            
            # Create session factory with proper configuration
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,  # Disable autoflush to prevent memory issues
                autocommit=False
            )
            
            self.logger.info(f"Database engine initialized with pool_size={engine_kwargs['pool_size']}, max_overflow={engine_kwargs['max_overflow']}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    @core_operation("get_database_session", max_retries=2)
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup."""
        session = None
        try:
            session = self._session_factory()
            self._connection_metrics["total_connections"] += 1
            self._connection_metrics["active_connections"] += 1
            
            # Test connection health
            await self._test_connection(session)
            
            yield session
            
            await session.commit()
            
        except Exception as error:
            if session:
                await session.rollback()
            self._connection_metrics["connection_errors"] += 1
            self.logger.error(f"Database session error: {error}")
            raise
        finally:
            if session:
                self._connection_metrics["active_connections"] -= 1
                await session.close()
    
    @asynccontextmanager
    async def session_context(self):
        """Context manager for database sessions."""
        async with self.get_session() as session:
            yield session
    
    async def _test_connection(self, session: AsyncSession):
        """Test database connection health."""
        try:
            # Simple health check query
            result = await session.execute(text("SELECT 1"))
            result.fetchone()
        except Exception as error:
            self._connection_metrics["failed_connections"] += 1
            raise DisconnectionError(f"Database connection test failed: {error}")
    
    @core_operation("initialize_database_tables")
    async def initialize_tables(self):
        """Initialize database tables."""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully")
        except Exception as error:
            self.logger.error(f"Failed to create database tables: {error}")
            raise
    
    @core_operation("close_database_connections")
    async def close_connections(self):
        """Close all database connections."""
        try:
            if self._engine:
                await self._engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as error:
            self.logger.error(f"Failed to close database connections: {error}")
            raise
    
    @core_operation("database_health_check")
    async def health_check(self):
        """Perform database health check."""
        start_time = datetime.utcnow()
        
        try:
            # Check if health check is needed
            if (self._last_health_check and 
                (start_time - self._last_health_check).total_seconds() < self._health_check_interval):
                return self.health_status
            
            # Perform health check
            async with self.session_context() as session:
                # Test basic connectivity
                await self._test_connection(session)
                
                # Check connection pool status
                pool_status = await self._get_pool_status()
                
                # Check for slow queries or connection issues
                error_rate = self._connection_metrics["connection_errors"] / max(1, self._connection_metrics["total_connections"])
                
                # Determine health status
                if error_rate > 0.1:  # More than 10% error rate
                    status = "degraded"
                elif pool_status["overflow"] > 0:
                    status = "degraded"
                else:
                    status = "healthy"
                
                end_time = datetime.utcnow()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                self.health_status = self.HealthStatus(
                    component_name=self.component_name,
                    status=status,
                    last_check=end_time,
                    response_time_ms=response_time_ms,
                    error_count=self._connection_metrics["connection_errors"],
                    details={
                        "pool_status": pool_status,
                        "connection_metrics": self._connection_metrics.copy(),
                        "error_rate": round(error_rate * 100, 2)
                    }
                )
                
                self._last_health_check = end_time
                return self.health_status
                
        except Exception as error:
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.health_status = self.HealthStatus(
                component_name=self.component_name,
                status="unhealthy",
                last_check=end_time,
                response_time_ms=response_time_ms,
                error_count=self.health_status.error_count + 1,
                details={"error": str(error)}
            )
            
            return self.health_status
    
    async def _get_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status."""
        if not self._engine:
            return {"error": "Database engine not initialized"}
        
        try:
            pool = self._engine.pool
            return {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
                "max_overflow": getattr(pool, '_max_overflow', 'unknown')
            }
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {"error": str(e)}
    
    async def monitor_connections(self) -> Dict[str, Any]:
        """Monitor database connections and perform cleanup if needed."""
        pool_status = await self._get_pool_status()
        
        # Check for connection leaks
        if isinstance(pool_status, dict) and "checked_out" in pool_status:
            checked_out = pool_status.get("checked_out", 0)
            pool_size = pool_status.get("pool_size", 0)
            
            # If more than 80% of connections are checked out, log a warning
            if pool_size > 0 and (checked_out / pool_size) > 0.8:
                self.logger.warning(
                    f"High connection usage detected: {checked_out}/{pool_size} "
                    f"connections checked out ({checked_out/pool_size*100:.1f}%)"
                )
            
            # If all connections are checked out, this might indicate a leak
            if checked_out >= pool_size:
                self.logger.error(
                    f"Potential connection leak detected: {checked_out}/{pool_size} "
                    "connections checked out"
                )
        
        return pool_status
    
    @core_operation("execute_database_query")
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query and return results."""
        start_time = datetime.utcnow()
        
        try:
            async with self.session_context() as session:
                result = await session.execute(text(query), params or {})
                
                # Convert result to list of dictionaries
                if result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    rows = []
                
                # Check for slow queries
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                if duration_ms > 1000:  # More than 1 second
                    self._connection_metrics["slow_queries"] += 1
                    self.logger.warning(f"Slow query detected: {duration_ms:.2f}ms - {query[:100]}...")
                
                return rows
                
        except Exception as error:
            self.logger.error(f"Query execution failed: {error}")
            raise
    
    @core_operation("database_migration")
    async def run_migration(self, migration_script: str):
        """Run a database migration script."""
        try:
            async with self.session_context() as session:
                # Split script into individual statements
                statements = [stmt.strip() for stmt in migration_script.split(';') if stmt.strip()]
                
                for statement in statements:
                    await session.execute(text(statement))
                
                await session.commit()
                self.logger.info("Database migration completed successfully")
                
        except Exception as error:
            self.logger.error(f"Database migration failed: {error}")
            raise
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get database connection metrics."""
        return {
            "component": self.component_name,
            "database_url": self.database_url.replace(self.database_url.split('@')[-1], '***') if '@' in self.database_url else self.database_url,
            "connection_metrics": self._connection_metrics.copy(),
            "pool_configuration": {
                "max_connections": self.max_connections,
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle
            }
        }
    
    async def reset_metrics(self):
        """Reset database metrics."""
        self._connection_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "connection_errors": 0,
            "slow_queries": 0
        }
        self.logger.info("Database metrics reset")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get database configuration."""
        config = super().get_configuration()
        config.update({
            "database_url": self.database_url.replace(self.database_url.split('@')[-1], '***') if '@' in self.database_url else self.database_url,
            "database_echo": self.database_echo,
            "max_connections": self.max_connections,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "health_check_interval": self._health_check_interval
        })
        return config

# Legacy compatibility functions
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Legacy function for backward compatibility."""
    async with database_manager.get_session() as session:
        yield session

async def init_db():
    """Legacy function for backward compatibility."""
    await database_manager.initialize_tables()

async def close_db():
    """Legacy function for backward compatibility."""
    await database_manager.close_connections()

# Global database manager instance
database_manager = DatabaseManager() 