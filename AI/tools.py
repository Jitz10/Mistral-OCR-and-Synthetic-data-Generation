import os
import json
import hashlib
import logging
import time
import asyncio
import redis
import numpy as np
import pandas as pd
import aiohttp
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps

# Check for required dependencies
try:
    import yfinance as yf
except ImportError as e:
    print("ERROR: Missing yfinance library. Please install it with:")
    print("pip install yfinance")
    raise e

try:
    import google.generativeai as genai
except ImportError as e:
    print("ERROR: Missing google-generativeai library. Please install it with:")
    print("pip install google-generativeai")
    raise e

try:
    from pymongo import MongoClient
    from bson import ObjectId
except ImportError as e:
    print("ERROR: Missing pymongo library. Please install it with:")
    print("pip install pymongo")
    raise e

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print("ERROR: Missing chromadb library. Please install it with:")
    print("pip install chromadb")
    raise e

try:
    from dotenv import load_dotenv
except ImportError as e:
    print("ERROR: Missing python-dotenv library. Please install it with:")
    print("pip install python-dotenv")
    raise e

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphasage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize API keys and connections
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Initialize Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client['alphasage']
    collection = db['alphasage_chunks']
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    mongo_client = None
    db = None
    collection = None

# Initialize Redis connection for caching
try:
    redis_client = redis.from_url(REDIS_URL)
    # Test the connection
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis connection failed, caching disabled: {e}")
    redis_client = None

# Initialize ChromaDB connection
try:
    chroma_client = chromadb.PersistentClient(
        path="./chromadb_data",
        settings=Settings(anonymized_telemetry=False)
    )
    chroma_collection = chroma_client.get_collection("alphasage_chunks")
    logger.info("ChromaDB connection established")
except Exception as e:
    logger.error(f"ChromaDB connection error: {e}")
    chroma_client = None
    chroma_collection = None

# --------------------------------
# Utility Functions
# --------------------------------

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log errors with context information for better debugging"""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg += f", Context: {json.dumps(context)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())

# Add in-memory cache as fallback
_memory_cache = {}
_cache_timestamps = {}

def cache_result(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Cache a result in Redis with TTL, fallback to memory cache
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized)
        ttl: Time to live in seconds (default: 1 hour)
        
    Returns:
        bool: Success status
    """
    if redis_client:
        try:
            serialized_value = json.dumps(value, default=str)
            redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")
    
    # Fallback to memory cache
    try:
        import time
        current_time = time.time()
        _memory_cache[key] = value
        _cache_timestamps[key] = current_time + ttl
        
        # Clean up expired entries (simple cleanup)
        if len(_memory_cache) > 100:  # Limit memory cache size
            expired_keys = [k for k, expiry in _cache_timestamps.items() if current_time > expiry]
            for k in expired_keys:
                _memory_cache.pop(k, None)
                _cache_timestamps.pop(k, None)
        
        logger.debug(f"Cached result in memory: {key}")
        return True
    except Exception as e:
        logger.warning(f"Memory cache write failed: {e}")
        return False

def get_cached_result(key: str) -> Optional[Any]:
    """
    Get a cached result from Redis or memory cache
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    if redis_client:
        try:
            cached_value = redis_client.get(key)
            if cached_value:
                return json.loads(cached_value)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")
    
    # Fallback to memory cache
    try:
        import time
        current_time = time.time()
        
        if key in _memory_cache and key in _cache_timestamps:
            if current_time <= _cache_timestamps[key]:
                logger.debug(f"Cache hit in memory: {key}")
                return _memory_cache[key]
            else:
                # Expired, remove from cache
                _memory_cache.pop(key, None)
                _cache_timestamps.pop(key, None)
        
        return None
    except Exception as e:
        logger.warning(f"Memory cache read failed: {e}")
        return None

# Add a utility function to check system health
def check_system_health() -> Dict[str, bool]:
    """
    Check the health of system dependencies
    
    Returns:
        Dict with status of each dependency
    """
    health = {
        "mongodb": False,
        "redis": False,
        "chromadb": False,
        "gemini_api": False
    }
    
    # Check MongoDB
    try:
        if mongo_client is not None:
            mongo_client.admin.command('ping')
            health["mongodb"] = True
    except Exception:
        pass
    
    # Check Redis
    try:
        if redis_client is not None:
            redis_client.ping()
            health["redis"] = True
    except Exception:
        pass
    
    # Check ChromaDB
    try:
        if chroma_client is not None and chroma_collection is not None:
            chroma_collection.count()
            health["chromadb"] = True
    except Exception:
        pass
    
    # Check Gemini API
    try:
        if GEMINI_API_KEY:
            health["gemini_api"] = True
    except Exception:
        pass
    
    return health

def generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate a deterministic cache key from function arguments
    
    Args:
        prefix: Key prefix
        **kwargs: Arguments to include in the key
        
    Returns:
        str: Cache key
    """
    # Sort kwargs for consistent ordering
    sorted_items = sorted(kwargs.items())
    
    # Serialize to JSON string for consistent representation
    args_str = json.dumps(sorted_items)
    
    # Create a hash to keep the key size manageable
    args_hash = hashlib.md5(args_str.encode()).hexdigest()
    
    return f"{prefix}:{args_hash}"

async def retry_async(func: Callable, *args, max_attempts: int = 3, **kwargs) -> Any:
    """
    Retry an async function with exponential backoff
    
    Args:
        func: Async function to retry
        *args: Arguments to pass to the function
        max_attempts: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    attempt = 0
    last_exception = None
    
    while attempt < max_attempts:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            last_exception = e
            
            if attempt >= max_attempts:
                log_error(e, {"action": "retry_async", "func": func.__name__, "attempt": attempt})
                break
            
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + (0.1 * np.random.random())
            logger.warning(f"Retry {attempt}/{max_attempts} for {func.__name__} after {wait_time:.2f}s: {str(e)}")
            await asyncio.sleep(wait_time)
    
    raise last_exception

def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbols for Indian equities
    
    Args:
        ticker: Ticker symbol (e.g., "TATAMOTORS", "TATAMOTORS.NS")
        
    Returns:
        Normalized ticker with .NS suffix if needed
    """
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    # Strip whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Handle BSE and NSE suffixes
    if ticker.endswith(".BO") or ticker.endswith(".NS"):
        return ticker
    
    # Default to NSE for Indian equities
    return f"{ticker}.NS"

def get_company_name_from_ticker(ticker: str) -> Optional[str]:
    """
    Get company name from ticker symbol using MongoDB or yfinance
    
    Args:
        ticker: Ticker symbol (e.g., "TATAMOTORS.NS")
        
    Returns:
        Company name or None if not found
    """
    # Common mappings for Indian companies
    common_mappings = {
        "TATAMOTORS.NS": "Tata Motors",
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys Limited",
        "WIPRO.NS": "Wipro Limited",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "BAJFINANCE.NS": "Bajaj Finance",
        "GANECOS.NS": "Ganesha Ecosphere"
    }
    
    # Try common mappings first
    if ticker in common_mappings:
        return common_mappings[ticker]
    
    # Try to find in MongoDB
    if collection:
        base_ticker = ticker.split('.')[0]
        try:
            # Try to find a chunk with similar company name
            regex_pattern = f".*{base_ticker}.*"
            result = collection.find_one({"company_name": {"$regex": regex_pattern, "$options": "i"}})
            if result and "company_name" in result:
                return result["company_name"]
        except Exception as e:
            log_error(e, {"action": "get_company_name_from_ticker", "ticker": ticker})
    
    # Fall back to yfinance
    try:
        ticker_info = yf.Ticker(ticker).info
        if ticker_info and "longName" in ticker_info:
            return ticker_info["longName"]
    except Exception as e:
        log_error(e, {"action": "get_company_name_from_ticker", "ticker": ticker})
    
    # Return a default based on the ticker if all else fails
    base_ticker = ticker.split('.')[0]
    return f"{base_ticker} Limited"

# --------------------------------
# Core Tool Classes
# --------------------------------

class ReasoningTool:
    """
    Tool for applying reasoning to financial data using Gemini 1.5 Flash model
    """
    
    @staticmethod
    async def reason_on_data(data: Union[str, Dict, List], query: str, 
                             max_words: int = 500, use_cache: bool = True) -> Dict[str, Any]:
        """
        Apply reasoning to financial data using Gemini 1.5 Flash
        
        Args:
            data: Financial data to reason on (text, dict, or list)
            query: Question or task for reasoning
            max_words: Maximum number of words in input (default: 500)
            use_cache: Whether to use Redis caching (default: True)
            
        Returns:
            Dict with reasoning, confidence, and sources
        """
        if not GEMINI_API_KEY:
            return {
                "reasoning": "Error: Gemini API key not configured",
                "confidence": 0.0,
                "sources": []
            }
        
        # Prepare data for reasoning
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, indent=2)
        else:
            data_str = str(data)
        
        # Limit input size to avoid token limits
        words = data_str.split()
        if len(words) > max_words:
            data_str = " ".join(words[:max_words]) + " [content truncated due to length]"
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = generate_cache_key("reasoning", data=data_str, query=query)
            cached_result = get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached reasoning result for query: {query[:50]}...")
                return cached_result
        
        try:
            # Use Gemini 1.5 Flash model for reasoning
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are a financial analyst specializing in Indian equities. Analyze the following data and answer the question.
            
            DATA:
            {data_str}
            
            QUESTION:
            {query}
            
            Your analysis should be concise, data-driven, and focused on the key insights. 
            Provide a confidence level (0.0-1.0) based on how well the data supports your reasoning.
            """
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 800}
            )
            
            # Process the response
            reasoning = response.text.strip()
            
            # Extract confidence level if present, or estimate based on certainty language
            confidence = 0.7  # Default confidence
            confidence_pattern = r"Confidence:?\s*(0\.\d+|1\.0)"
            import re
            confidence_match = re.search(confidence_pattern, reasoning)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                # Remove the confidence line from the reasoning
                reasoning = re.sub(confidence_pattern, "", reasoning).strip()
            
            result = {
                "reasoning": reasoning,
                "confidence": confidence,
                "sources": ["gemini-1.5-flash"]
            }
            
            # Cache the result
            if cache_key:
                cache_result(cache_key, result, ttl=3600)  # Cache for 1 hour
            
            return result
            
        except Exception as e:
            error_message = str(e)
            
            # Provide more helpful responses for specific error types
            if "RATE_LIMIT_EXCEEDED" in error_message or "quota" in error_message.lower():
                logger.warning(f"Gemini API rate limit exceeded: {error_message[:100]}...")
                
                # Do basic analysis if rate limited
                if isinstance(data, dict) and "metrics" in data:
                    metrics = data.get("metrics", [])
                    company = data.get("company", "the company")
                    
                    # Simple fallback analysis based on common financial metrics
                    fallback_analysis = f"Based on the metrics provided for {company}, "
                    
                    positive_indicators = 0
                    total_indicators = len(metrics)
                    
                    for metric in metrics:
                        name = metric.get("name", "").lower()
                        value = metric.get("value", "")
                        
                        # Convert percentage strings to numbers if possible
                        if isinstance(value, str) and "%" in value:
                            try:
                                value = float(value.replace("%", ""))
                            except:
                                pass
                        
                        # Simple rules for common metrics
                        if name in ["profit margin", "operating margin"] and isinstance(value, (int, float)) and value > 5:
                            positive_indicators += 1
                        elif name in ["roe", "roa"] and isinstance(value, (int, float)) and value > 10:
                            positive_indicators += 1
                        elif name in ["revenue growth", "growth"] and isinstance(value, (int, float)) and value > 10:
                            positive_indicators += 1
                        elif "p/e" in name and isinstance(value, (int, float)) and 5 < value < 25:
                            positive_indicators += 1
                    
                    # Calculate confidence based on available metrics
                    confidence = round(positive_indicators / max(total_indicators, 1), 1) if total_indicators > 0 else 0.5
                    
                    if positive_indicators > total_indicators / 2:
                        fallback_analysis += "the financial performance appears relatively strong."
                    else:
                        fallback_analysis += "the financial performance shows mixed or concerning signals."
                    
                    fallback_analysis += " (Note: This is a fallback analysis due to Gemini API rate limits)"
                    
                    return {
                        "reasoning": fallback_analysis,
                        "confidence": confidence,
                        "sources": ["fallback analysis (rate limited)"]
                    }
                
                return {
                    "reasoning": "Unable to provide analysis due to Gemini API rate limits. Please try again later.",
                    "confidence": 0.0,
                    "sources": ["error: rate limited"]
                }
            
            log_error(e, {"action": "reason_on_data", "query": query})
            return {
                "reasoning": f"Error in reasoning: {error_message}",
                "confidence": 0.0,
                "sources": []
            }

class YFinanceNumberTool:
    """
    Tool for fetching financial data and metrics from yfinance
    """

    @staticmethod
    def get_financial_ratio_by_date(company_ticker: str, ratio_name: str, target_date: str, duration: str = '2y') -> Dict[str, Any]:
        """
        Get financial ratio for a company as of a specific date.
        """
        import warnings
        warnings.filterwarnings('ignore')
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        import traceback

        TODAY = datetime.now()
        TODAY_STR = TODAY.strftime("%Y-%m-%d")

        try:
            if not company_ticker:
                return {"error": "Company ticker is required"}
            if not ratio_name:
                return {"error": "Ratio name is required"}
            if not target_date:
                return {"error": "Target date is required"}
            try:
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            except ValueError:
                return {"error": "Invalid date format. Use YYYY-MM-DD (e.g., '2025-12-31')"}
            
            if target_dt > TODAY:
                return {"error": f"Target date cannot be in the future. Today is {TODAY_STR}"}
                #return {"error": f"Target date cannot be in the future. Today is {TODAY_STR}"}
            days_from_today = (TODAY - target_dt).days
            available_ratios = {
                'P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'EV/Revenue', 'EV/EBITDA',
                'Profit Margin', 'Gross Margin', 'Return on Assets (ROA)', 'Return on Equity (ROE)',
                'Debt-to-Equity', 'Debt-to-Assets', 'Current Ratio', 'Quick Ratio',
                'Dividend Rate', 'Dividend Yield', 'Payout Ratio', 'Beta', 'Short Ratio',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Stock Price'
            }
            if ratio_name not in available_ratios:
                return {
                    "error": f"Ratio '{ratio_name}' not available",
                    "available_ratios": sorted(list(available_ratios))
                }
            end_date = min(target_dt + timedelta(days=30), TODAY)
            if duration.endswith('y'):
                years = int(duration[:-1])
                start_date = target_dt - timedelta(days=years * 365)
            elif duration.endswith('m'):
                months = int(duration[:-1])
                start_date = target_dt - timedelta(days=months * 30)
            elif duration.endswith('d'):
                days = int(duration[:-1])
                start_date = target_dt - timedelta(days=days)
            else:
                start_date = target_dt - timedelta(days=365)
            try:
                ticker = yf.Ticker(company_ticker.upper())
                hist_data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=False)
            except Exception as e:
                return {"error": f"Failed to fetch data for {company_ticker}: {str(e)}"}
            if hist_data.empty:
                return {"error": f"No historical data found for {company_ticker} around {target_date}"}
            hist_data.index = pd.to_datetime(hist_data.index)
            if hist_data.index.tz is not None:
                target_dt_tz = target_dt.replace(tzinfo=hist_data.index.tz)
            else:
                target_dt_tz = target_dt
            date_diffs = abs(hist_data.index - target_dt_tz)
            closest_date_idx = date_diffs.argmin()
            closest_date = hist_data.index[closest_date_idx]
            closest_row = hist_data.iloc[closest_date_idx]
            try:
                try:
                    current_info = ticker.info
                except Exception as info_exc:
                    logger.warning(f"yfinance .info failed for {company_ticker}: {info_exc}")
                    current_info = {}
                if not current_info or len(current_info) < 5:
                    current_info = {}
            except Exception:
                current_info = {}
            financials = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()
            try:
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
            except Exception as e:
                pass
            ratio_value = None
            stock_price = closest_row['Close']
            if ratio_name == 'P/E Ratio':
                eps = current_info.get('trailingEps') or current_info.get('forwardEps')
                ratio_value = stock_price / eps if eps and eps > 0 else None
            elif ratio_name == 'P/S Ratio':
                revenue = current_info.get('totalRevenue')
                shares = current_info.get('sharesOutstanding')
                if revenue and shares and revenue > 0:
                    ratio_value = stock_price / (revenue / shares)
            elif ratio_name == 'P/B Ratio':
                book_value = current_info.get('bookValue')
                ratio_value = stock_price / book_value if book_value and book_value > 0 else None
            elif ratio_name in ['Profit Margin', 'Gross Margin', 'Return on Assets (ROA)', 'Return on Equity (ROE)',
                               'Current Ratio', 'Quick Ratio', 'Dividend Rate', 'Dividend Yield', 'Payout Ratio',
                               'Beta', 'Short Ratio']:
                ratio_mapping = {
                    'Profit Margin': 'profitMargins',
                    'Gross Margin': 'grossMargins',
                    'Return on Assets (ROA)': 'returnOnAssets',
                    'Return on Equity (ROE)': 'returnOnEquity',
                    'Current Ratio': 'currentRatio',
                    'Quick Ratio': 'quickRatio',
                    'Dividend Rate': 'dividendRate',
                    'Dividend Yield': 'dividendYield',
                    'Payout Ratio': 'payoutRatio',
                    'Beta': 'beta',
                    'Short Ratio': 'shortRatio'
                }
                ratio_value = current_info.get(ratio_mapping.get(ratio_name))
            elif ratio_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                ratio_value = closest_row[ratio_name]
            elif ratio_name == 'Stock Price':
                ratio_value = stock_price
            if hasattr(closest_date, 'tz') and closest_date.tz:
                closest_date_naive = closest_date.tz_localize(None)
            else:
                closest_date_naive = closest_date
            days_difference = abs((closest_date_naive - target_dt).days)
            context_data = {}
            closest_idx = hist_data.index.get_loc(closest_date)
            start_idx = max(0, closest_idx - 5)
            end_idx = min(len(hist_data), closest_idx + 6)
            context_range = hist_data.iloc[start_idx:end_idx]
            for date, row in context_range.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                if ratio_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    context_data[date_str] = float(row[ratio_name])
                elif ratio_name == 'Stock Price':
                    context_data[date_str] = float(row['Close'])
            result = {
                "ticker": company_ticker.upper(),
                "company_name": current_info.get('longName', 'N/A'),
                "ratio_name": ratio_name,
                "target_date": target_date,
                "closest_available_date": closest_date.strftime('%Y-%m-%d'),
                "days_difference": days_difference,
                "days_from_today": days_from_today,
                "today": TODAY_STR,
                "ratio_value": float(ratio_value) if ratio_value is not None else None,
                "stock_price_on_date": float(stock_price),
                "context_data": context_data,
                "date_context": {
                    "is_recent": days_from_today <= 30,
                    "is_current_year": target_dt.year == TODAY.year,
                    "is_weekend_request": target_dt.weekday() >= 5,
                    "market_status": "Market closed (weekend)" if target_dt.weekday() >= 5 else "Market trading day"
                },
                "data_quality": {
                    "has_historical_price": True,
                    "has_company_info": bool(current_info),
                    "has_historical_financials": not financials.empty,
                    "has_historical_balance_sheet": not balance_sheet.empty,
                    "data_freshness": "good" if days_difference <= 7 else "moderate" if days_difference <= 30 else "stale"
                },
                "note": f"Data retrieved for {company_ticker.upper()} as of {closest_date.strftime('%Y-%m-%d')} "
                       f"({days_difference} days from target date, {days_from_today} days ago from today)"
            }
            return result
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "traceback": traceback.format_exc()[-1000:]
            }

    @staticmethod
    async def fetch_financial_data(
        ticker: str,
        metric: str,
        period: str = "5y",
        frequency: str = "quarterly",
        use_cache: bool = True,
        target_date: Optional[str] = None,
        duration: str = "2y"
    ) -> Dict[str, Any]:
        """
        Fetch financial metrics for a company using yfinance.
        If target_date is provided, fetch ratio as of that date.
        """
        # If target_date is provided, use the point-in-time ratio fetcher
        if target_date:
            return YFinanceNumberTool.get_financial_ratio_by_date(
                company_ticker=ticker,
                ratio_name=metric,
                target_date=target_date,
                duration=duration
            )
        # If target_date is not provided, proceed with regular data fetching
        try:
            # Validate ticker
            ticker = validate_ticker(ticker)
            
            # Generate cache key if caching is enabled
            cache_key = None
            if use_cache:
                cache_key = generate_cache_key("yfinance_data", ticker=ticker, metric=metric, 
                                               period=period, frequency=frequency)
                cached_result = get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Using cached financial data for {ticker}, metric: {metric}")
                    return cached_result
            
            # Normalize metric name for lookup
            metric_lower = metric.lower()
            
            # Map common metric names to yfinance attributes
            metric_mapping = {
                "p/e": "trailingPE",
                "pe": "trailingPE",
                "pe ratio": "trailingPE",
                "forward p/e": "forwardPE",
                "forward pe": "forwardPE",
                "p/b": "priceToBook",
                "pb": "priceToBook",
                "price/book": "priceToBook",
                "ev/ebitda": "enterpriseToEbitda",
                "ev/revenue": "enterpriseToRevenue",
                "dividend yield": "dividendYield",
                "roe": "returnOnEquity",
                "roa": "returnOnAssets",
                "profit margin": "profitMargins",
                "operating margin": "operatingMargins",
                "beta": "beta",
                "52 week high": "fiftyTwoWeekHigh",
                "52 week low": "fiftyTwoWeekLow",
                "market cap": "marketCap",
                "shares outstanding": "sharesOutstanding",
                "eps": "trailingEps",
                "forward eps": "forwardEps",
                "book value": "bookValue",
                "debt to equity": "debtToEquity",
                "revenue": "totalRevenue",
                "net income": "netIncomeToCommon",
                "ebitda": "ebitda"
            }
            
            # Fetch data from yfinance
            ticker_obj = yf.Ticker(ticker)
            
            # For financial statements and time series data
            financial_statements = {
                "income statement": "income_stmt",
                "balance sheet": "balance_sheet",
                "cash flow": "cash_flow",
                "quarterly income statement": "quarterly_income_stmt",
                "quarterly balance sheet": "quarterly_balance_sheet",
                "quarterly cash flow": "quarterly_cash_flow"
            }
            
            result = {"metric": metric, "data": [], "sources": ["yfinance"]}
            
            # Check if this is a financial statement request
            if metric_lower in financial_statements:
                statement_type = financial_statements[metric_lower]
                statement_data = getattr(ticker_obj, statement_type)
                
                if statement_data is not None and not statement_data.empty:
                    # Convert to appropriate format
                    data_points = []
                    for column in statement_data.columns:
                        for index, value in statement_data[column].items():
                            if pd.notnull(value):
                                data_points.append({
                                    "date": index.strftime("%Y-%m-%d"),
                                    "metric": column,
                                    "value": float(value) if isinstance(value, (int, float)) else None
                                })
                    
                    result["data"] = data_points
                    
                    # Cache the result
                    if cache_key:
                        cache_result(cache_key, result, ttl=3600)
                    
                    return result
            
            # Fix the strftime error - handle cases where index might be a string
            # Add this code where processing historical data:
            
            # Check if index is a string and convert it to datetime if needed
            from datetime import datetime
            def safe_format_date(date_obj):
                """Safely format a date object or string to YYYY-MM-DD format"""
                if isinstance(date_obj, str):
                    # Try to parse the string to a datetime
                    try:
                        date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        # If parsing fails, return the string as is
                        return date_obj
                else:
                    # If it's already a datetime object, format it
                    return date_obj.strftime("%Y-%m-%d")
            
            # Then replace code like:
            # "date": index.strftime("%Y-%m-%d"),
            # With:
            # "date": safe_format_date(index),
            
            # For historical data
            if metric_lower in ["price", "close", "open", "high", "low", "volume"]:
                hist_data = ticker_obj.history(period=period)
                
                if not hist_data.empty:
                    column = metric.capitalize() if metric_lower != "price" else "Close"
                    data_points = []
                    
                    for date, row in hist_data.iterrows():
                        if column in row and pd.notnull(row[column]):
                            data_points.append({
                                "date": safe_format_date(date),
                                "value": float(row[column])
                            })
                    
                    result["data"] = data_points
                    
                    # Cache the result
                    if cache_key:
                        cache_result(cache_key, result, ttl=3600)
                    
                    return result
            
            # For point-in-time metrics from info
            if metric_lower in metric_mapping:
                try:
                    info = ticker_obj.info
                except Exception as info_exc:
                    logger.warning(f"yfinance .info failed for {ticker}: {info_exc}")
                    info = {}
                yf_metric = metric_mapping[metric_lower]
                if yf_metric in info and info[yf_metric] is not None:
                    value = info[yf_metric]
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    result["data"] = [{"date": current_date, "value": value}]
                    
                    # Cache the result
                    if cache_key:
                        cache_result(cache_key, result, ttl=3600)
                    
                    return result
            
            # Fallback to MongoDB if yfinance doesn't have the data
            if collection is not None:
                company_name = get_company_name_from_ticker(ticker)
                
                if company_name:
                    # Find relevant chunks with this metric
                    query = {
                        "company_name": company_name,
                        "$or": [
                            {"category": "Valuation Ratios"},
                            {"category": "Technical Ratios"}
                        ]
                    }
                    
                    chunks = list(collection.find(query).limit(5))
                    metric_data = []
                    
                    for chunk in chunks:
                        content = chunk.get("content", {})
                        
                        # For table data
                        if content.get("type") == "table":
                            table_data = content.get("table", [])
                            for cell in table_data:
                                cell_value = cell.get("value", "").lower()
                                if metric_lower in cell_value:
                                    # Look for neighboring cells with numbers
                                    for neighbor in table_data:
                                        if neighbor != cell:
                                            try:
                                                value_text = neighbor.get("value", "")
                                                value = float(''.join(c for c in value_text if c.isdigit() or c in '.-'))
                                                date = chunk.get("document_date", "")
                                                metric_data.append({"date": date, "value": value})
                                            except:
                                                continue
                        
                        # For text data
                        elif content.get("type") == "text":
                            text = content.get("text", "")
                            if metric_lower in text.lower():
                                # Simple regex to find numbers near the metric mention
                                import re
                                pattern = f"{metric}[\\s:]*([\\d.]+)"
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                
                                if matches:
                                    for match in matches:
                                        try:
                                            value = float(match)
                                            date = chunk.get("document_date", "")
                                            metric_data.append({"date": date, "value": value})
                                        except:
                                            continue
                    
                    if metric_data:
                        result["data"] = metric_data
                        result["sources"].append("mongodb")
                        
                        # Cache the result
                        if cache_key:
                            cache_result(cache_key, result, ttl=3600)
                        
                        return result
            
            # Return empty result if no data found
            return {
                "metric": metric,
                "data": [],
                "sources": ["No data found"],
                "error": f"No data found for {metric} in {ticker}"
            }
            
        except Exception as e:
            log_error(e, {"action": "fetch_financial_data", "ticker": ticker, "metric": metric})
            return {
                "metric": metric,
                "data": [],
                "sources": [],
                "error": str(e)
            }

class YFinanceNewsTool:
    """
    Tool for fetching company news from yfinance
    """
    
    @staticmethod
    async def fetch_company_news(ticker: str, max_results: int = 10, 
                                use_cache: bool = True, store_in_mongodb: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a company using yfinance
        
        Args:
            ticker: Stock ticker symbol (e.g., "TATAMOTORS.NS")
            max_results: Maximum number of news articles to return
            use_cache: Whether to use Redis caching
            store_in_mongodb: Whether to store news in MongoDB
            
        Returns:
            List of news articles with title, date, url, summary, and source
        """
        try:
            # Validate ticker
            ticker = validate_ticker(ticker)
            
            # Generate cache key if caching is enabled
            cache_key = None
            if use_cache:
                cache_key = generate_cache_key("yfinance_news", ticker=ticker, max_results=max_results)
                cached_result = get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Using cached news for {ticker}")
                    return cached_result
            
            # Fetch news from yfinance
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            if not news:
                return []
            
            # Process news articles
            articles = []
            for article in news[:max_results]:
                news_item = {
                    "title": article.get("title", ""),
                    "date": datetime.fromtimestamp(article.get("providerPublishTime", 0)).strftime("%Y-%m-%d"),
                    "url": article.get("link", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("publisher", "yfinance")
                }
                articles.append(news_item)
            
            # Store in MongoDB if requested
            if store_in_mongodb and collection is not None and len(articles) > 0:
                company_name = get_company_name_from_ticker(ticker)
                
                if company_name:
                    # Batch store news articles
                    news_chunks = []
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    
                    for article in articles:
                        chunk = {
                            "company_name": company_name,
                            "document_date": article["date"],
                            "source": "News",
                            "category": "Company Disclosures",
                            "content": {
                                "type": "text",
                                "text": f"{article['title']}. {article['summary']}",
                                "url": article["url"],
                                "keywords": []
                            },
                            "created_at": current_date,
                            "chunk_id": f"{company_name}_news_{hashlib.md5(article['title'].encode()).hexdigest()}"
                        }
                        news_chunks.append(chunk)
                    
                    if news_chunks:
                        try:
                            collection.insert_many(news_chunks, ordered=False)
                            logger.info(f"Stored {len(news_chunks)} news articles for {company_name} in MongoDB")
                        except Exception as e:
                            logger.warning(f"Failed to store news in MongoDB: {e}")

            # Cache the result
            if cache_key:
                cache_result(cache_key, articles, ttl=1800)  # Cache for 30 minutes
            
            return articles
            
        except Exception as e:
            log_error(e, {"action": "fetch_company_news", "ticker": ticker})
            return []

class ArithmeticCalculationTool:
    """
    Tool for performing calculations on financial data
    """
    
    @staticmethod
    def calculate_metrics(inputs: Dict[str, Union[float, List[Dict[str, float]]]], 
                         formula: str) -> Dict[str, Any]:
        """
        Calculate financial metrics using provided inputs and formula
        
        Args:
            inputs: Dictionary of input values or time series data
            formula: Formula to calculate (e.g., "ROE = Net Income / Shareholders Equity")
            
        Returns:
            Dict with calculated results and sources
        """
        try:
            # Parse the formula to identify variables
            formula_parts = formula.split('=')
            if len(formula_parts) != 2:
                return {"error": "Invalid formula format. Use 'Metric = Formula'"}
            
            metric_name = formula_parts[0].strip()
            formula_expr = formula_parts[1].strip()
            
            # Extract variable names from the formula
            import re
            variables = re.findall(r'[A-Za-z_][A-Za-z0-9_\s]*', formula_expr)
            variables = [var.strip() for var in variables if var.strip() and var.strip().lower() not in ['and', 'or', 'not', 'if', 'else', 'for', 'while']]
            
            # Check if all required variables are provided
            missing_vars = [var for var in variables if var not in inputs]
            if missing_vars:
                return {
                    "error": f"Missing required variables: {', '.join(missing_vars)}",
                    "results": [],
                    "sources": []
                }
            
            # Check if inputs are time series data
            time_series = False
            for key, value in inputs.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    time_series = True
                    break
            
            if time_series:
                # For time series data, calculate for each time point
                # First, organize data by date
                dates = set()
                for key, value in inputs.items():
                    if isinstance(value, list):
                        for point in value:
                            if isinstance(point, dict) and 'date' in point:
                                dates.add(point['date'])
                
                results = []
                for date in sorted(dates):
                    date_inputs = {}
                    for key, value in inputs.items():
                        if isinstance(value, list):
                            for point in value:
                                if isinstance(point, dict) and point.get('date') == date and 'value' in point:
                                    date_inputs[key] = point['value']
                        else:
                            date_inputs[key] = value
                    
                    # Skip if missing any inputs for this date
                    if len(date_inputs) < len(variables):
                        continue
                    
                    # Create the evaluation environment
                    eval_env = {k: v for k, v in date_inputs.items()}
                    
                    # Calculate using eval (controlled environment)
                    formula_to_eval = formula_expr
                    for var in variables:
                        if ' ' in var:  # Replace spaces with underscores for evaluation
                            formula_to_eval = formula_to_eval.replace(var, var.replace(' ', '_'))
                            if var in eval_env:
                                eval_env[var.replace(' ', '_')] = eval_env.pop(var)
                    
                    try:
                        # Use numpy for more complex calculations
                        for func in ['sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'mean', 'median', 'std', 'min', 'max']:
                            if func in formula_to_eval:
                                eval_env[func] = getattr(np, func)
                        
                        result = eval(formula_to_eval, {"__builtins__": {}}, eval_env)
                        results.append({"date": date, "metric": metric_name, "value": float(result)})
                    except ZeroDivisionError:
                        logger.warning(f"Division by zero in calculation for date {date}")
                        results.append({"date": date, "metric": metric_name, "error": "Division by zero"})
                    except Exception as calc_error:
                        logger.warning(f"Calculation error for date {date}: {calc_error}")
                
                return {
                    "results": results,
                    "sources": ["calculated"]
                }
            else:
                # For point values, calculate once
                # Create the evaluation environment
                eval_env = {k: v for k, v in inputs.items()}
                
                # Calculate using eval (controlled environment)
                formula_to_eval = formula_expr
                for var in variables:
                    if ' ' in var:  # Replace spaces with underscores for evaluation
                        formula_to_eval = formula_to_eval.replace(var, var.replace(' ', '_'))
                        if var in eval_env:
                            eval_env[var.replace(' ', '_')] = eval_env.pop(var)
                
                # Use numpy for more complex calculations
                for func in ['sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'mean', 'median', 'std', 'min', 'max']:
                    if func in formula_to_eval:
                        eval_env[func] = getattr(np, func)
                
                result = eval(formula_to_eval, {"__builtins__": {}}, eval_env)
                
                return {
                    "results": [{"metric": metric_name, "value": float(result)}],
                    "sources": ["calculated"]
                }
                
        except Exception as e:
            log_error(e, {"action": "calculate_metrics", "formula": formula})
            return {
                "error": str(e),
                "results": [],
                "sources": []
            }

class VectorSearchRAGTool:
    """
    Tool for searching knowledge base using vector similarity
    """
    
    @staticmethod
    async def search_knowledge_base(query: str, company_name: Optional[str] = None, 
                                   filters: Dict[str, Any] = None, 
                                   max_results: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Search vector database for relevant chunks using query
        
        Args:
            query: Text query for semantic search
            company_name: Optional company name to filter results
            filters: Additional filters (e.g., category, date range)
            max_results: Maximum number of results to return
            use_cache: Whether to use Redis caching
            
        Returns:
            List of relevant chunks from the knowledge base
        """
        if not chroma_collection:
            logger.error("ChromaDB not initialized")
            return []
        
        try:
            # Generate cache key if caching is enabled
            cache_key = None
            if use_cache:
                cache_key = generate_cache_key("vector_search", query=query, company_name=company_name, 
                                               filters=filters, max_results=max_results)
                cached_result = get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Using cached vector search result for query: {query[:50]}...")
                    return cached_result
            
            # Build ChromaDB-compatible where clause
            where_clause = {}
            
            # Add company name filter if provided
            if company_name:
                where_clause["company_name"] = {"$eq": company_name}
            
            # Add category filter if provided in filters
            if filters and 'category' in filters:
                category = filters['category']
                if where_clause:
                    # ChromaDB requires using $and operator for multiple conditions
                    where_clause = {
                        "$and": [
                            {"company_name": {"$eq": company_name}} if company_name else {"company_name": {"$ne": ""}},
                            {"category": {"$eq": category}}
                        ]
                    }
                else:
                    where_clause["category"] = {"$eq": category}
            
            # Only add where if it is non-empty and not just {}
            try:
                query_params = {
                    "query_texts": [query],
                    "n_results": max_results,
                    "include": ["metadatas", "documents", "distances"]
                }
                if where_clause and (("$and" in where_clause and where_clause["$and"]) or (not "$and" in where_clause and where_clause)):
                    query_params["where"] = where_clause
                chroma_results = chroma_collection.query(**query_params)
                
                # Convert ChromaDB results to our format
                results = []
                if chroma_results and 'documents' in chroma_results and chroma_results['documents']:
                    documents = chroma_results['documents'][0]
                    metadatas = chroma_results.get('metadatas', [[]])[0]
                    distances = chroma_results.get('distances', [[]])[0]
                    
                    for i, doc in enumerate(documents):
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        distance = distances[i] if i < len(distances) else 1.0
                        
                        result = {
                            "content": {
                                "type": "text",
                                "text": doc,
                                "metadata": metadata
                            },
                            "company_name": metadata.get("company_name", ""),
                            "category": metadata.get("category", ""),
                            "document_date": metadata.get("document_date", ""),
                            "source": metadata.get("source", ""),
                            "similarity_score": 1.0 - distance,  # Convert distance to similarity
                            "chunk_id": metadata.get("chunk_id", f"chunk_{i}")
                        }
                        results.append(result)
                
                # Cache the result
                if cache_key and results:
                    cache_result(cache_key, results, ttl=3600)
                
                return results
                
            except Exception as chroma_error:
                # Handle ChromaDB embedding dimension error and empty where error
                err_msg = str(chroma_error)
                if "embedding with dimension" in err_msg or "Expected where to have exactly one operator" in err_msg:
                    logger.error(f"ChromaDB query error: {err_msg}")
                    return []
                logger.error(f"ChromaDB direct query failed: {chroma_error}")
                
                # Fallback to vector.py module if available
                try:
                    # Import the function from vector.py module
                    import sys
                    import os
                    
                    # Add the parent directory to the sys.path
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.append(current_dir)
                    
                    # Import the AlphaSageVectorDB class
                    from vector import AlphaSageVectorDB
                    
                    # Initialize the vector DB
                    vector_db = AlphaSageVectorDB()
                    
                    # Prepare parameters for the vector DB
                    start_date = None
                    end_date = None
                    category = None
                    
                    if filters:
                        if 'start_date' in filters:
                            start_date = filters['start_date']
                        if 'end_date' in filters:
                            end_date = filters['end_date']
                        if 'category' in filters:
                            category = filters['category']
                    
                    # Use the retrieve_chunks method with modified approach
                    # Since ChromaDB has issues with complex filters, we'll query without filters
                    # and filter the results manually
                    all_results = vector_db.retrieve_chunks(
                        company_name=None,  # Don't filter in ChromaDB
                        category=None,      # Don't filter in ChromaDB  
                        start_date=start_date,
                        end_date=end_date,
                        query_text=query,
                        n_results=max_results * 3  # Get more results to filter manually
                    )
                    
                    # Manual filtering
                    filtered_results = []
                    for result in all_results:
                        # Check company name filter
                        if company_name and result.get("company_name", "").lower() != company_name.lower():
                            continue
                        
                        # Check category filter
                        if category and result.get("category", "").lower() != category.lower():
                            continue
                        
                        filtered_results.append(result)
                        
                        # Stop when we have enough results
                        if len(filtered_results) >= max_results:
                            break
                    
                    # Close the vector DB connection to free resources
                    vector_db.close()
                    
                    results = filtered_results
                    
                except ImportError as import_error:
                    logger.error(f"Failed to import vector module: {import_error}")
                    results = []
                except Exception as vector_error:
                    logger.error(f"Vector search fallback failed: {vector_error}")
                    results = []

            # Cache the result
            if cache_key and results:
                cache_result(cache_key, results, ttl=3600)
            
            return results
            
        except Exception as e:
            log_error(e, {"action": "search_knowledge_base", "query": query, "company_name": company_name})
            return []

# --------------------------------
# Enhanced Testing Functions
# --------------------------------

async def test_tool_comprehensive(tool_name: str, input_data: Dict[str, Any], expected: Any = None, test_name: str = "") -> Dict[str, Any]:
    """
    Comprehensive test a tool with given input and expected output, including performance metrics
    
    Args:
        tool_name: Name of the tool to test
        input_data: Input data for the tool
        expected: Expected output (optional)
        test_name: Descriptive name for the test
        
    Returns:
        Dict with detailed test results
    """
    test_id = f"{tool_name}_{test_name}" if test_name else tool_name
    logger.info(f"Testing tool: {test_id}")
    
    start_time = time.time()
    result = None
    error = None
    memory_before = None
    memory_after = None
    
    try:
        # Measure memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
        
        if tool_name == "reasoning":
            data = input_data.get("data", "")
            query = input_data.get("query", "")
            max_words = input_data.get("max_words", 500)
            use_cache = input_data.get("use_cache", True)
            result = await ReasoningTool.reason_on_data(data, query, max_words, use_cache)
            
        elif tool_name == "financial_data":
            ticker = input_data.get("ticker", "")
            metric = input_data.get("metric", "")
            period = input_data.get("period", "5y")
            frequency = input_data.get("frequency", "quarterly")
            target_date = input_data.get("target_date")
            duration = input_data.get("duration", "2y")
            result = await YFinanceNumberTool.fetch_financial_data(
                ticker, metric, period, frequency, target_date=target_date, duration=duration
            )
            
        elif tool_name == "company_news":
            ticker = input_data.get("ticker", "")
            max_results = input_data.get("max_results", 10)
            use_cache = input_data.get("use_cache", True)
            store_in_mongodb = input_data.get("store_in_mongodb", False)  # Don't store during tests
            result = await YFinanceNewsTool.fetch_company_news(ticker, max_results, use_cache, store_in_mongodb)
            
        elif tool_name == "calculate_metrics":
            inputs = input_data.get("inputs", {})
            formula = input_data.get("formula", "")
            result = ArithmeticCalculationTool.calculate_metrics(inputs, formula)
            
        elif tool_name == "search_knowledge_base":
            query = input_data.get("query", "")
            company_name = input_data.get("company_name", None)
            filters = input_data.get("filters", {})
            max_results = input_data.get("max_results", 5)
            use_cache = input_data.get("use_cache", True)
            result = await VectorSearchRAGTool.search_knowledge_base(query, company_name, filters, max_results, use_cache)
            
        elif tool_name == "financial_ratio_by_date":
            ticker = input_data.get("ticker", "")
            ratio_name = input_data.get("ratio_name", "")
            target_date = input_data.get("target_date", "")
            duration = input_data.get("duration", "2y")
            result = YFinanceNumberTool.get_financial_ratio_by_date(ticker, ratio_name, target_date, duration)
            
        else:
            error = f"Unknown tool: {tool_name}"
        
        # Measure memory after execution
        if memory_before:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
    except Exception as e:
        error = str(e)
        log_error(e, {"action": "test_tool_comprehensive", "tool": tool_name, "input": input_data})
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    test_result = {
        "test_id": test_id,
        "tool": tool_name,
        "test_name": test_name,
        "input": input_data,
        "output": result,
        "execution_time": execution_time,
        "success": error is None,
        "performance": {
            "execution_time_ms": execution_time * 1000,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_delta_mb": (memory_after - memory_before) if (memory_before and memory_after) else None
        }
    }
    
    if error:
        test_result["error"] = error
    
    # Enhanced validation
    if expected is not None and result is not None:
        validation_result = validate_test_result(result, expected, tool_name)
        test_result["validation"] = validation_result
    
    # Tool-specific analysis
    if result and not error:
        analysis = analyze_tool_output(tool_name, result, input_data)
        test_result["analysis"] = analysis
    
    logger.info(f"Test complete: {test_id}, Success: {test_result['success']}, Time: {execution_time:.3f}s")
    return test_result

def validate_test_result(result: Any, expected: Any, tool_name: str) -> Dict[str, Any]:
    """Enhanced validation for test results"""
    validation = {
        "passed": False,
        "details": {},
        "score": 0.0
    }
    
    try:
        if tool_name == "reasoning":
            # Validate reasoning tool output
            if isinstance(result, dict):
                required_keys = ["reasoning", "confidence", "sources"]
                missing_keys = [k for k in required_keys if k not in result]
                validation["details"]["missing_keys"] = missing_keys
                validation["details"]["has_reasoning"] = bool(result.get("reasoning"))
                validation["details"]["confidence_valid"] = 0.0 <= result.get("confidence", -1) <= 1.0
                validation["score"] = 1.0 - (len(missing_keys) / len(required_keys))
                validation["passed"] = len(missing_keys) == 0 and validation["details"]["confidence_valid"]
        
        elif tool_name == "financial_data":
            # Validate financial data output
            if isinstance(result, dict):
                has_data = bool(result.get("data"))
                has_sources = bool(result.get("sources"))
                has_metric = bool(result.get("metric"))
                validation["details"]["has_data"] = has_data
                validation["details"]["has_sources"] = has_sources
                validation["details"]["has_metric"] = has_metric
                validation["details"]["data_count"] = len(result.get("data", []))
                validation["score"] = sum([has_data, has_sources, has_metric]) / 3.0
                validation["passed"] = has_metric and has_sources
        
        elif tool_name == "company_news":
            # Validate news output
            if isinstance(result, list):
                validation["details"]["news_count"] = len(result)
                valid_articles = 0
                for article in result:
                    if isinstance(article, dict) and all(k in article for k in ["title", "date", "source"]):
                        valid_articles += 1
                validation["details"]["valid_articles"] = valid_articles
                validation["score"] = valid_articles / max(len(result), 1)
                validation["passed"] = valid_articles > 0
        
        elif tool_name == "calculate_metrics":
            # Validate calculation output
            if isinstance(result, dict):
                has_results = bool(result.get("results"))
                has_sources = bool(result.get("sources"))
                no_error = "error" not in result
                validation["details"]["has_results"] = has_results
                validation["details"]["has_sources"] = has_sources
                validation["details"]["no_error"] = no_error
                validation["score"] = sum([has_results, has_sources, no_error]) / 3.0
                validation["passed"] = has_results and no_error
        
        elif tool_name == "search_knowledge_base":
            # Validate vector search output
            if isinstance(result, list):
                validation["details"]["result_count"] = len(result)
                valid_chunks = 0
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        valid_chunks += 1
                validation["details"]["valid_chunks"] = valid_chunks
                validation["score"] = valid_chunks / max(len(result), 1) if result else 0.0
                validation["passed"] = valid_chunks > 0
        
        # Compare with expected if provided
        if expected is not None:
            if isinstance(expected, dict) and isinstance(result, dict):
                matching_keys = sum(1 for k, v in expected.items() if result.get(k) == v)
                validation["details"]["expected_matches"] = matching_keys
                validation["details"]["expected_total"] = len(expected)
            else:
                validation["details"]["exact_match"] = result == expected
                
    except Exception as e:
        validation["error"] = str(e)
        validation["passed"] = False
        validation["score"] = 0.0
    
    return validation

def analyze_tool_output(tool_name: str, result: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze tool output for insights and quality metrics"""
    analysis = {}
    
    try:
        if tool_name == "reasoning":
            if isinstance(result, dict):
                reasoning_text = result.get("reasoning", "")
                analysis["reasoning_length"] = len(reasoning_text)
                analysis["confidence_level"] = result.get("confidence", 0.0)
                analysis["has_fallback"] = "fallback" in reasoning_text.lower()
                analysis["quality_score"] = min(1.0, len(reasoning_text) / 100) * result.get("confidence", 0.0)
        
        elif tool_name == "financial_data":
            if isinstance(result, dict):
                data_points = result.get("data", [])
                analysis["data_points_count"] = len(data_points)
                analysis["sources_count"] = len(result.get("sources", []))
                analysis["has_error"] = "error" in result
                analysis["data_completeness"] = 1.0 if data_points else 0.0
        
        elif tool_name == "company_news":
            if isinstance(result, list):
                analysis["articles_count"] = len(result)
                analysis["avg_summary_length"] = sum(len(a.get("summary", "")) for a in result) / max(len(result), 1)
                analysis["unique_sources"] = len(set(a.get("source", "") for a in result))
                analysis["coverage_score"] = min(1.0, len(result) / input_data.get("max_results", 10))
        
        elif tool_name == "calculate_metrics":
            if isinstance(result, dict):
                results_list = result.get("results", [])
                analysis["calculations_count"] = len(results_list)
                analysis["has_time_series"] = any("date" in r for r in results_list if isinstance(r, dict))
                analysis["calculation_success"] = "error" not in result
        
        elif tool_name == "search_knowledge_base":
            if isinstance(result, list):
                analysis["chunks_count"] = len(result)
                analysis["avg_similarity"] = sum(r.get("similarity_score", 0) for r in result) / max(len(result), 1)
                analysis["companies_covered"] = len(set(r.get("company_name", "") for r in result))
                analysis["relevance_score"] = analysis["avg_similarity"]
                
    except Exception as e:
        analysis["analysis_error"] = str(e)
    
    return analysis

async def run_comprehensive_test_suite():
    """Run comprehensive test suite with extensive coverage"""
    print("\n" + "="*80)
    print("ALPHASAGE TOOLS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check system health first
    health = check_system_health()
    print(f"\n=== System Health Check ===")
    for service, status in health.items():
        status_text = "✓ ONLINE" if status else "✗ OFFLINE"
        print(f"{service.upper():15}: {status_text}")
    
    print(f"\n=== Test Configuration ===")
    print(f"MongoDB Available: {health['mongodb']}")
    print(f"Redis Available: {health['redis']}")
    print(f"ChromaDB Available: {health['chromadb']}")
    print(f"Gemini API Available: {health['gemini_api']}")
    
    all_test_results = []
    
    # 1. YFinanceNumberTool Tests
    print(f"\n{'='*60}")
    print("1. YFINANCE NUMBER TOOL TESTS")
    print(f"{'='*60}")
    
    yfinance_tests = [
        {
            "name": "basic_pe_ratio",
            "input": {"ticker": "TATAMOTORS.NS", "metric": "P/E"},
            "description": "Basic P/E ratio fetch"
        },
        {
            "name": "historical_price",
            "input": {"ticker": "RELIANCE.NS", "metric": "price", "period": "1y"},
            "description": "Historical price data"
        },
        {
            "name": "invalid_ticker",
            "input": {"ticker": "INVALIDTICKER", "metric": "P/E"},
            "description": "Invalid ticker handling"
        },
        {
            "name": "ratio_by_date",
            "input": {"ticker": "TATAMOTORS.NS", "metric": "Stock Price", "target_date": "2024-01-15"},
            "description": "Point-in-time ratio fetch"
        },
        {
            "name": "future_date_error",
            "input": {"ticker": "RELIANCE.NS", "metric": "P/E Ratio", "target_date": "2026-01-01"},
            "description": "Future date error handling"
        }
    ]
    
    for test_config in yfinance_tests:
        print(f"\nRunning: {test_config['description']}")
        result = await test_tool_comprehensive("financial_data", test_config["input"], test_name=test_config["name"])
        all_test_results.append(result)
        print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
              f"({result['execution_time']:.3f}s)")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # 2. Financial Ratio by Date Tests (specific method)
    print(f"\n{'='*60}")
    print("2. FINANCIAL RATIO BY DATE TESTS")
    print(f"{'='*60}")
    
    ratio_tests = [
        {
            "name": "pe_ratio_past_date",
            "input": {"ticker": "TATAMOTORS.NS", "ratio_name": "P/E Ratio", "target_date": "2024-06-01"},
            "description": "P/E ratio for past date"
        },
        {
            "name": "stock_price_weekend",
            "input": {"ticker": "RELIANCE.NS", "ratio_name": "Stock Price", "target_date": "2024-06-02"},  # Sunday
            "description": "Stock price on weekend"
        },
        {
            "name": "invalid_ratio_name",
            "input": {"ticker": "TATAMOTORS.NS", "ratio_name": "Invalid Ratio", "target_date": "2024-06-01"},
            "description": "Invalid ratio name handling"
        },
        {
            "name": "missing_ticker",
            "input": {"ticker": "", "ratio_name": "P/E Ratio", "target_date": "2024-06-01"},
            "description": "Missing ticker error"
        }
    ]
    
    for test_config in ratio_tests:
        print(f"\nRunning: {test_config['description']}")
        result = await test_tool_comprehensive("financial_ratio_by_date", test_config["input"], test_name=test_config["name"])
        all_test_results.append(result)
        print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
              f"({result['execution_time']:.3f}s)")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # 3. News Tool Tests
    print(f"\n{'='*60}")
    print("3. COMPANY NEWS TOOL TESTS")
    print(f"{'='*60}")
    
    news_tests = [
        {
            "name": "basic_news_fetch",
            "input": {"ticker": "TATAMOTORS.NS", "max_results": 5},
            "description": "Basic news fetch"
        },
        {
            "name": "large_news_fetch",
            "input": {"ticker": "RELIANCE.NS", "max_results": 20},
            "description": "Large news fetch"
        },
        {
            "name": "invalid_ticker_news",
            "input": {"ticker": "INVALIDTICKER.NS", "max_results": 5},
            "description": "Invalid ticker news fetch"
        },
        {
            "name": "zero_results",
            "input": {"ticker": "TATAMOTORS.NS", "max_results": 0},
            "description": "Zero results request"
        }
    ]
    
    for test_config in news_tests:
        print(f"\nRunning: {test_config['description']}")
        result = await test_tool_comprehensive("company_news", test_config["input"], test_name=test_config["name"])
        all_test_results.append(result)
        print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
              f"({result['execution_time']:.3f}s)")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
        elif result['output']:
            print(f"Articles found: {len(result['output'])}")
    
    # 4. Arithmetic Calculation Tests
    print(f"\n{'='*60}")
    print("4. ARITHMETIC CALCULATION TOOL TESTS")
    print(f"{'='*60}")
    
    calc_tests = [
        {
            "name": "simple_roe_calculation",
            "input": {
                "inputs": {"Net_Income": 1000, "Shareholders_Equity": 5000},
                "formula": "ROE = Net_Income / Shareholders_Equity"
            },
            "expected": {"results": [{"metric": "ROE", "value": 0.2}]},
            "description": "Simple ROE calculation"
        },
        {
            "name": "complex_calculation",
            "input": {
                "inputs": {"Revenue": 10000, "Costs": 7000, "Assets": 50000},
                "formula": "Profit_Margin = (Revenue - Costs) / Revenue"
            },
            "description": "Complex profit margin calculation"
        },
        {
            "name": "time_series_calculation",
            "input": {
                "inputs": {
                    "Revenue": [
                        {"date": "2023-01-01", "value": 1000},
                        {"date": "2023-02-01", "value": 1100}
                    ],
                    "Costs": [
                        {"date": "2023-01-01", "value": 800},
                        {"date": "2023-02-01", "value": 850}
                    ]
                },
                "formula": "Profit = Revenue - Costs"
            },
            "description": "Time series calculation"
        },
        {
            "name": "invalid_formula",
            "input": {
                "inputs": {"A": 10, "B": 5},
                "formula": "Invalid Formula Format"
            },
            "description": "Invalid formula format"
        },
        {
            "name": "missing_variables",
            "input": {
                "inputs": {"A": 10},
                "formula": "Result = A + B"
            },
            "description": "Missing variables error"
        },
        {
            "name": "division_by_zero",
            "input": {
                "inputs": {"A": 10, "B": 0},
                "formula": "Result = A / B"
            },
            "description": "Division by zero handling"
        }
    ]
    
    for test_config in calc_tests:
        print(f"\nRunning: {test_config['description']}")
        result = await test_tool_comprehensive("calculate_metrics", test_config["input"], 
                                             expected=test_config.get("expected"), 
                                             test_name=test_config["name"])
        all_test_results.append(result)
        print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
              f"({result['execution_time']:.3f}s)")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
        elif result.get('validation'):
            print(f"Validation: {'✓ PASS' if result['validation']['passed'] else '✗ FAIL'}")
    
    # 5. Reasoning Tool Tests
    if health["gemini_api"]:
        print(f"\n{'='*60}")
        print("5. REASONING TOOL TESTS")
        print(f"{'='*60}")
        
        reasoning_tests = [
            {
                "name": "financial_analysis",
                "input": {
                    "data": {
                        "company": "Tata Motors",
                        "metrics": [
                            {"name": "P/E", "value": 12.5},
                            {"name": "Revenue Growth", "value": "15%"},
                            {"name": "Profit Margin", "value": "8.2%"}
                        ]
                    },
                    "query": "Analyze the financial health of this company"
                },
                "description": "Financial health analysis"
            },
            {
                "name": "investment_recommendation",
                "input": {
                    "data": {
                        "company": "Reliance Industries",
                        "metrics": [
                            {"name": "P/E", "value": 25.3},
                            {"name": "ROE", "value": 12.8},
                            {"name": "Debt/Equity", "value": 0.45}
                        ]
                    },
                    "query": "Should I invest in this company?"
                },
                "description": "Investment recommendation"
            },
            {
                "name": "large_data_input",
                "input": {
                    "data": " ".join(["This is a large text input."] * 200),  # Large input
                    "query": "Summarize the key points",
                    "max_words": 100
                },
                "description": "Large data input handling"
            },
            {
                "name": "empty_data",
                "input": {
                    "data": "",
                    "query": "What can you tell me about this?"
                },
                "description": "Empty data handling"
            }
        ]
        
        for test_config in reasoning_tests:
            print(f"\nRunning: {test_config['description']}")
            # Add delay to avoid rate limits
            await asyncio.sleep(1)
            result = await test_tool_comprehensive("reasoning", test_config["input"], test_name=test_config["name"])
            all_test_results.append(result)
            print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
                  f"({result['execution_time']:.3f}s)")
            if not result['success']:
                print(f"Error: {result.get('error', 'Unknown error')}")



    else:
        print(f"\n{'='*60}")
        print("5. REASONING TOOL TESTS - SKIPPED (API not available)")
        print(f"{'='*60}")
    
    # 6. Vector Search Tests
    if health["chromadb"]:
        print(f"\n{'='*60}")
        print("6. VECTOR SEARCH TOOL TESTS")
        print(f"{'='*60}")
        
        vector_tests = [
            {
                "name": "basic_search",
                "input": {
                    "query": "financial performance",
                    "max_results": 3
                },
                "description": "Basic vector search"
            },
            {
                "name": "company_filtered_search",
                "input": {
                    "query": "growth prospects",
                    "company_name": "Ganesha Ecosphere",
                    "max_results": 5
                },
                "description": "Company-filtered search"
            },
            {
                "name": "category_filtered_search",
                "input": {
                    "query": "revenue analysis",
                    "filters": {"category": "Financial Metrics"},
                    "max_results": 3
                },
                "description": "Category-filtered search"
            },
            {
                "name": "empty_query",
                "input": {
                    "query": "",
                    "max_results": 5
                },
                "description": "Empty query handling"
            },
            {
                "name": "no_results_query",
                "input": {
                    "query": "completely unrelated quantum physics topic",
                    "max_results": 5
                },
                "description": "No results query"
            }
        ]
        
        for test_config in vector_tests:
            print(f"\nRunning: {test_config['description']}")
            result = await test_tool_comprehensive("search_knowledge_base", test_config["input"], test_name=test_config["name"])
            all_test_results.append(result)
            print(f"Status: {'✓ PASS' if result['success'] else '✗ FAIL'} "
                  f"({result['execution_time']:.3f}s)")
            if not result['success']:
                print(f"Error: {result.get('error', 'Unknown error')}")
            elif result['output']:
                print(f"Results found: {len(result['output'])}")
                if result['output']:
                    avg_similarity = sum(r.get('similarity_score', 0) for r in result['output']) / len(result['output'])
                    print(f"Average similarity: {avg_similarity:.3f}")
    else:
        print(f"\n{'='*60}")
        print("6. VECTOR SEARCH TOOL TESTS - SKIPPED (ChromaDB not available)")
        print(f"{'='*60}")
    
    # Generate comprehensive test report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    # Overall statistics
    total_tests = len(all_test_results)
    passed_tests = sum(1 for r in all_test_results if r['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"\n=== Overall Statistics ===")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} (✓)")
    print(f"Failed: {failed_tests} (✗)")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Performance statistics
    execution_times = [r['execution_time'] for r in all_test_results]
    print(f"\n=== Performance Statistics ===")
    print(f"Total Execution Time: {sum(execution_times):.3f}s")
    print(f"Average Test Time: {sum(execution_times)/len(execution_times):.3f}s")
    print(f"Fastest Test: {min(execution_times):.3f}s")
    print(f"Slowest Test: {max(execution_times):.3f}s")
    
    # Tool-specific statistics
    tool_stats = {}
    for result in all_test_results:
        tool = result['tool']
        if tool not in tool_stats:
            tool_stats[tool] = {'total': 0, 'passed': 0, 'times': []}
        tool_stats[tool]['total'] += 1
        if result['success']:
            tool_stats[tool]['passed'] += 1
        tool_stats[tool]['times'].append(result['execution_time'])
    
    print(f"\n=== Tool-Specific Results ===")
    for tool, stats in tool_stats.items():
        success_rate = (stats['passed'] / stats['total']) * 100
        avg_time = sum(stats['times']) / len(stats['times'])
        print(f"{tool:25}: {stats['passed']}/{stats['total']} "
              f"({success_rate:5.1f}%) avg: {avg_time:.3f}s")
    
    # Failed tests details
    failed_results = [r for r in all_test_results if not r['success']]
    if failed_results:
        print(f"\n=== Failed Tests Details ===")
        for result in failed_results:
            print(f"❌ {result['test_id']}: {result.get('error', 'Unknown error')}")
    
    # Quality metrics
    validation_scores = [r.get('validation', {}).get('score', 0.0) for r in all_test_results if r.get('validation')]
    if validation_scores:
        print(f"\n=== Quality Metrics ===")
        print(f"Average Validation Score: {sum(validation_scores)/len(validation_scores):.3f}")
        print(f"High Quality Tests (>0.8): {sum(1 for s in validation_scores if s > 0.8)}")
        print(f"Low Quality Tests (<0.5): {sum(1 for s in validation_scores if s < 0.5)}")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if failed_tests > 0:
        print("• Review failed tests and fix underlying issues")
    if not health["mongodb"]:
        print("• Consider setting up MongoDB for enhanced functionality")
    if not health["redis"]:
        print("• Consider setting up Redis for improved caching")
    if not health["gemini_api"]:
        print("• Add Gemini API key for reasoning capabilities")
    if max(execution_times) > 10:
        print("• Optimize slow-running tests for better performance")
    
    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETED")
    print(f"{'='*80}")
    
    return all_test_results

# --------------------------------
# Main Function
# --------------------------------

async def main():
    """Main function for comprehensive testing"""
    print("AlphaSage Financial Analysis Tools")
    print("Comprehensive Test Suite")
    print("Date: June 9, 2025")
    print("=" * 50)
    
    # Run comprehensive test suite
    test_results = await run_comprehensive_test_suite()
    
    # Optionally save results to file
    try:
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in test_results:
            serializable_result = result.copy()lue in serializable_result.items():
            # Convert any non-serializable objects to strings                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):














asyncio.run(main())# Run the main function        print(f"Error saving test results: {e}")    except Exception as e:        print(f"Test results saved to {filename}")                    json.dump(serializable_results, f, indent=2)        with open(filename, "w") as f:                    serializable_results.append(serializable_result)                serializable_result["output"] = json.dumps(serializable_result["output"], default=str)            if isinstance(serializable_result.get("output"), (dict, list)):                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_tests": len(test_results),
                "passed_tests": sum(1 for r in test_results if r['success']),
                "results": serializable_results
            }, f, indent=2)
        
        print(f"\nTest results saved to: {filename}")
        
    except Exception as e:
        print(f"\nWarning: Could not save test results to file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
