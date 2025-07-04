import logging
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger("quota_manager")

@dataclass
class QuotaUsage:
    tokens_used: int = 0
    requests_made: int = 0
    last_reset: datetime = datetime.utcnow()
    last_request: datetime = datetime.utcnow()

class GeminiQuotaManager:
    def __init__(
        self,
        daily_token_limit: int = 10_000_000,
        requests_per_minute: int = 120,
        persist_file: Optional[str] = None
    ):
        self.daily_token_limit = daily_token_limit
        self.requests_per_minute = requests_per_minute
        self.persist_file = persist_file or "gemini_quota.json"
        self.usage: Dict[str, QuotaUsage] = {}
        self._lock = asyncio.Lock()
        self._load_usage()
        
        # Start background task for quota persistence
        asyncio.create_task(self._periodic_save())
    
    def _load_usage(self):
        """Load quota usage from persistent storage."""
        try:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'r') as f:
                    data = json.load(f)
                    for repo, usage in data.items():
                        self.usage[repo] = QuotaUsage(
                            tokens_used=usage['tokens_used'],
                            requests_made=usage['requests_made'],
                            last_reset=datetime.fromisoformat(usage['last_reset']),
                            last_request=datetime.fromisoformat(usage['last_request'])
                        )
                logger.info(f"Loaded quota usage for {len(self.usage)} repositories")
        except Exception as e:
            logger.error(f"Error loading quota usage: {e}")
    
    async def _save_usage(self):
        """Save quota usage to persistent storage."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
            
            data = {
                repo: {
                    'tokens_used': usage.tokens_used,
                    'requests_made': usage.requests_made,
                    'last_reset': usage.last_reset.isoformat(),
                    'last_request': usage.last_request.isoformat()
                }
                for repo, usage in self.usage.items()
            }
            
            # Write to temporary file first
            temp_file = f"{self.persist_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.persist_file)
            logger.debug("Saved quota usage")
        except Exception as e:
            logger.error(f"Error saving quota usage: {e}")
    
    async def _periodic_save(self):
        """Periodically save quota usage."""
        while True:
            await asyncio.sleep(300)  # Save every 5 minutes
            await self._save_usage()
    
    def _reset_if_needed(self, repo: str):
        """Reset quota if daily limit has expired."""
        if repo not in self.usage:
            self.usage[repo] = QuotaUsage()
            return
        
        usage = self.usage[repo]
        now = datetime.utcnow()
        
        # Reset daily quota if 24 hours have passed
        if now - usage.last_reset >= timedelta(days=1):
            usage.tokens_used = 0
            usage.last_reset = now
            logger.info(f"Reset daily quota for {repo}")
        
        # Reset request count after 1 minute
        if now - usage.last_request >= timedelta(minutes=1):
            usage.requests_made = 0
    
    async def check_quota(self, repo: str) -> bool:
        """Check if quota is available."""
        async with self._lock:
            self._reset_if_needed(repo)
            usage = self.usage[repo]
            
            # Check daily token limit
            if usage.tokens_used >= self.daily_token_limit:
                logger.warning(f"Daily token limit exceeded for {repo}")
                return False
            
            # Check rate limit
            if usage.requests_made >= self.requests_per_minute:
                logger.warning(f"Request rate limit exceeded for {repo}")
                return False
            
            return True
    
    async def update_usage(self, repo: str, tokens_used: int):
        """Update quota usage after a request."""
        async with self._lock:
            self._reset_if_needed(repo)
            usage = self.usage[repo]
            
            usage.tokens_used += tokens_used
            usage.requests_made += 1
            usage.last_request = datetime.utcnow()
            
            logger.debug(
                f"Updated quota for {repo}: "
                f"{usage.tokens_used}/{self.daily_token_limit} tokens, "
                f"{usage.requests_made}/{self.requests_per_minute} requests/min"
            )
    
    async def get_usage_stats(self, repo: str) -> Dict:
        """Get current usage statistics."""
        async with self._lock:
            self._reset_if_needed(repo)
            usage = self.usage[repo]
            
            return {
                'tokens_used_today': usage.tokens_used,
                'tokens_remaining': self.daily_token_limit - usage.tokens_used,
                'requests_this_minute': usage.requests_made,
                'requests_remaining': self.requests_per_minute - usage.requests_made,
                'last_reset': usage.last_reset.isoformat(),
                'last_request': usage.last_request.isoformat()
            }

# Global instance
quota_manager = GeminiQuotaManager(
    persist_file=os.path.join(
        os.getenv('QUOTA_PERSIST_DIR', 'data'),
        'gemini_quota.json'
    )
) 