import os
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import aiofiles

logger = logging.getLogger("quota_manager")

class GeminiQuotaManager:
    def __init__(self, quota_file_path: str = "gemini_quota.json"):
        self.quota_file_path = os.path.abspath(quota_file_path)
        self.daily_quota = 60000  # Default daily token quota
        self.usage_data: Dict[str, Dict] = {}
        self._ensure_quota_file_exists()

    def _ensure_quota_file_exists(self):
        """Ensure the quota file exists and is properly initialized."""
        os.makedirs(os.path.dirname(self.quota_file_path), exist_ok=True)
        if not os.path.exists(self.quota_file_path):
            with open(self.quota_file_path, 'w') as f:
                json.dump({}, f)

    async def _load_usage_data(self):
        """Load usage data from file."""
        try:
            async with aiofiles.open(self.quota_file_path, 'r') as f:
                content = await f.read()
                self.usage_data = json.loads(content) if content else {}
        except Exception as e:
            logger.error(f"Error loading quota data: {e}")
            self.usage_data = {}

    async def _save_usage_data(self):
        """Save usage data to file."""
        try:
            async with aiofiles.open(self.quota_file_path, 'w') as f:
                await f.write(json.dumps(self.usage_data, indent=2))
        except Exception as e:
            logger.error(f"Error saving quota usage: {e}")

    def _clean_old_data(self, repo_data: Dict):
        """Remove usage data older than 24 hours."""
        cutoff_time = (datetime.utcnow() - timedelta(days=1)).isoformat()
        repo_data['usage_history'] = [
            entry for entry in repo_data.get('usage_history', [])
            if entry['timestamp'] > cutoff_time
        ]

    async def check_quota(self, repo_full_name: str) -> bool:
        """Check if the repository has remaining quota."""
        await self._load_usage_data()
        
        if repo_full_name not in self.usage_data:
            return True

        repo_data = self.usage_data[repo_full_name]
        self._clean_old_data(repo_data)
        
        # Calculate total tokens used in the last 24 hours
        total_tokens = sum(
            entry['tokens'] for entry in repo_data.get('usage_history', [])
        )
        
        return total_tokens < self.daily_quota

    async def update_usage(self, repo_full_name: str, tokens_used: int):
        """Update token usage for a repository."""
        await self._load_usage_data()
        
        if repo_full_name not in self.usage_data:
            self.usage_data[repo_full_name] = {'usage_history': []}
        
        repo_data = self.usage_data[repo_full_name]
        self._clean_old_data(repo_data)
        
        # Add new usage entry
        repo_data['usage_history'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'tokens': tokens_used
        })
        
        await self._save_usage_data()

    async def get_usage_stats(self, repo_full_name: str) -> Dict:
        """Get usage statistics for a repository."""
        await self._load_usage_data()
        
        if repo_full_name not in self.usage_data:
            return {
                'tokens_used_today': 0,
                'quota_remaining': self.daily_quota,
                'quota_reset_in': '24 hours'
            }
        
        repo_data = self.usage_data[repo_full_name]
        self._clean_old_data(repo_data)
        
        tokens_used = sum(
            entry['tokens'] for entry in repo_data.get('usage_history', [])
        )
        
        # Find the oldest usage timestamp to calculate reset time
        usage_history = repo_data.get('usage_history', [])
        if usage_history:
            oldest_timestamp = datetime.fromisoformat(usage_history[0]['timestamp'])
            reset_time = oldest_timestamp + timedelta(days=1)
            reset_in = reset_time - datetime.utcnow()
            reset_hours = round(reset_in.total_seconds() / 3600, 1)
            quota_reset_in = f"{reset_hours} hours"
        else:
            quota_reset_in = "24 hours"
        
        return {
            'tokens_used_today': tokens_used,
            'quota_remaining': max(0, self.daily_quota - tokens_used),
            'quota_reset_in': quota_reset_in
        }

# Global instance
quota_manager = GeminiQuotaManager() 