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

# Load environment variables
load_dotenv()

# Database Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# API Configuration
GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1", "")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3", "")
GEMINI_BASE_DELAY = float(os.getenv("GEMINI_BASE_DELAY", "1.0"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RATE_LIMIT = int(os.getenv("GEMINI_RATE_LIMIT", "60"))

# Directory Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
LOG_DIR = os.getenv("LOG_DIR", "./logs")
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")

# PDF Configuration
PDF_PAGE_SIZE = os.getenv("PDF_PAGE_SIZE", "letter")
PDF_MARGINS = {
    "top": float(os.getenv("PDF_MARGIN_TOP", "72")),
    "bottom": float(os.getenv("PDF_MARGIN_BOTTOM", "72")),
    "left": float(os.getenv("PDF_MARGIN_LEFT", "72")),
    "right": float(os.getenv("PDF_MARGIN_RIGHT", "72"))
}
PDF_FONT_SIZE = int(os.getenv("PDF_FONT_SIZE", "12"))
PDF_LINE_SPACING = float(os.getenv("PDF_LINE_SPACING", "1.2"))

# Analysis Configuration
ANALYSIS_THREADS = int(os.getenv("ANALYSIS_THREADS", "4"))
ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT", "300"))
DATA_QUALITY_THRESHOLD = float(os.getenv("DATA_QUALITY_THRESHOLD", "0.7"))

# Vector Search Configuration
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))
VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.7"))

# Error Reporting and Monitoring
ERROR_REPORTING_EMAIL = os.getenv("ERROR_REPORTING_EMAIL", "")
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "false").lower() == "true"

# Security
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

# Default Values
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "GANECOS.NS")
DEFAULT_COMPANY_NAME = os.getenv("DEFAULT_COMPANY_NAME", "Ganesha Ecosphere Limited")

# Create required directories
for directory in [OUTPUT_DIR, CACHE_DIR, LOG_DIR, TEMP_DIR, CHROMA_PERSIST_DIR]:
    os.makedirs(directory, exist_ok=True)
