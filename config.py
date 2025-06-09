import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 10):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def can_make_call(self) -> bool:
        """Check if we can make another API call"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        return len(self.calls) < self.calls_per_minute
    
    def record_call(self):
        """Record that an API call was made"""
        self.calls.append(time.time())
    
    def wait_time(self) -> float:
        """Get the time to wait before next call"""
        if self.can_make_call():
            return 0.0
        
        oldest_call = min(self.calls)
        return 60.0 - (time.time() - oldest_call)

# Global rate limiter for Gemini API
GEMINI_RATE_LIMITER = RateLimiter(calls_per_minute=8)  # Conservative limit

# Enhanced LLM configuration with rate limiting
def get_llm_config_with_rate_limit() -> Dict[str, Any]:
    """Get LLM configuration with rate limiting support"""
    return {
        "config_list": [{
            "model": "gemini-1.5-flash",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "api_type": "google"
        }],
        "temperature": 0.1,  # Lower temperature for more consistent results
        "max_tokens": 1500,   # Reduced token limit
        "cache_seed": 42,
        "timeout": 30,        # 30 second timeout
        "retry_wait_time": 2, # Wait 2 seconds between retries
        "max_retries": 2      # Only 2 retries to avoid rate limits
    }

# Database configurations
DATABASE_CONFIG = {
    "mongodb": {
        "uri": os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
        "database": "alphasage",
        "timeout": 5000
    },
    "redis": {
        "url": os.getenv('REDIS_URL', 'redis://localhost:6379'),
        "timeout": 5
    },
    "chromadb": {
        "persist_directory": os.getenv('CHROMADB_PATH', './chroma_db'),
        "embedding_model": "all-MiniLM-L6-v2"  # 384 dimensions
    }
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "cache_ttl": {
        "business_analysis": 7200,    # 2 hours
        "sector_analysis": 3600,      # 1 hour
        "current_affairs": 1800,      # 30 minutes
        "financial_data": 3600,       # 1 hour
        "predictions": 7200           # 2 hours
    },
    "concurrent_limit": 3,            # Max 3 concurrent analyses
    "timeout_seconds": 300,           # 5 minute timeout per agent
    "retry_attempts": 2               # Max 2 retry attempts
}
