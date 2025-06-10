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
import re
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from autogen import ConversableAgent
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import yfinance as yf
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Enhanced logging configuration for DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphasage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('alphasage.tools')

# Initialize API keys and connections
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMMA_API_URL = os.getenv('GEMMA_API_URL', 'http://127.0.0.1:1234')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Constants
DB_NAME = "alphasage_chunks"
COLLECTION_NAME = "chunks"
REDIS_EXPIRY = 3600  # 1 hour
GEMINI_RATE_LIMIT = 2  # seconds between API calls
GEMINI_MAX_RETRIES = 3
GEMINI_BASE_DELAY = 4  # Base delay for exponential backoff

# Fixed: Initialize Gemini API with round-robin key rotation and enhanced error handling
class RoundRobinGeminiAPI:
    """Manages round-robin access to multiple Gemini API keys with rate limiting."""
    
    def __init__(self):
        """Initialize the API key manager."""
        self.logger = logging.getLogger('alphasage.gemini')
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.rate_limit_delay = float(os.getenv('GEMINI_BASE_DELAY', '4.0'))
        self.max_retries = int(os.getenv('GEMINI_MAX_RETRIES', '3'))
        
    def _load_api_keys(self) -> List[str]:
        """Load Gemini API keys from environment variables."""
        keys = []
        i = 1
        while True:
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if not key:
                break
            keys.append(key)
            i += 1
            
        # Fallback to single key if numbered keys not found
        if not keys:
            single_key = os.getenv('GEMINI_API_KEY')
            if single_key:
                keys.append(single_key)
        
        if not keys:
            self.logger.error("No Gemini API keys found in environment variables")
            raise ValueError("No Gemini API keys found in environment variables")
            
        self.logger.debug(f"Loaded {len(keys)} Gemini API keys")
        return keys
        
    def _get_next_key(self) -> str:
        """Get the next API key in round-robin fashion."""
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.logger.debug(f"Using API key {self.current_key_index}: {key[:8]}...")
        return key
        
    async def generate_content(self, prompt: str, max_retries: int = None) -> str:
        """Generate content using Gemini API with rate limiting and retries."""
        self.logger.debug(f"Generating content with prompt length: {len(prompt)}")
        
        if max_retries is None:
            max_retries = self.max_retries
            
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last_request)
                
                # Get API key and configure client
                api_key = self._get_next_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Make API call
                self.logger.debug(f"Making Gemini API call (attempt {retry_count + 1}/{max_retries + 1})")
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 1024,
                    }
                )
                
                self.last_request_time = time.time()
                
                if response and response.text:
                    self.logger.debug(f"Successfully generated content with length: {len(response.text)}")
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "429" in error_msg or "RATE_LIMIT_EXCEEDED" in error_msg:
                    delay = self.rate_limit_delay * (2 ** (retry_count - 1))
                    self.logger.warning(f"Rate limit exceeded, retrying in {delay}s (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Error in Gemini API call: {error_msg}")
                    if retry_count > max_retries:
                        raise
                    await asyncio.sleep(self.rate_limit_delay)
                    
        raise Exception(f"Failed to generate content after {max_retries} retries")

# Initialize Gemini API instance
gemini_api = RoundRobinGeminiAPI()

# Fixed: Enhanced MongoDB connection with retry logic
def _init_mongodb_with_retry(max_retries: int = 3, delay: int = 5) -> Optional[MongoClient]:
    """Initialize MongoDB connection with retry logic to handle _OperationCancelled."""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})")
            mongo_client = MongoClient(
                os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
                serverSelectionTimeoutMS=30000,  # 30 seconds
                connectTimeoutMS=20000,           # 20 seconds
                socketTimeoutMS=20000,            # 20 seconds
                maxPoolSize=10
            )
            # Test the connection
            mongo_client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            return mongo_client
        except Exception as e:
            logger.warning(f"MongoDB connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.debug(f"Retrying MongoDB connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"MongoDB connection failed after {max_retries} attempts")
                return None

# Initialize MongoDB connection with retry
try:
    mongo_client = _init_mongodb_with_retry()
    if mongo_client:
        db = mongo_client['alphasage']
        collection = db['alphasage_chunks']
        logger.info("MongoDB collections initialized")
    else:
        mongo_client = None
        db = None
        collection = None
except Exception as e:
    logger.error(f"MongoDB initialization error: {e}")
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

# Add Gemma API client
async def get_gemma_client():
    """Get Gemma API client with retry logic"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{GEMMA_API_URL}/v1/models") as response:
                if response.status == 200:
                    return session
    except Exception as e:
        logger.warning(f"Failed to connect to Gemma API: {e}")
    return None

# Add connection management functions
def get_chromadb_connection():
    """Get ChromaDB connection with error handling"""
    try:
        if chroma_client is None:
            return None
        return chroma_client
    except Exception as e:
        logger.error(f"ChromaDB connection error: {e}")
        return None

def get_mongodb_connection():
    """Get MongoDB connection with error handling"""
    try:
        if mongo_client is None:
            return None
        return mongo_client
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        return None

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

# Fixed: Enhanced caching functions with proper async support
async def cache_result(cache_key: str, data: Dict, ttl: int = 3600) -> bool:
    """Cache result in Redis with enhanced error handling."""
    try:
        if redis_client:
            logger.debug(f"Caching result with key: {cache_key}, TTL: {ttl}")
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data, default=str)
            )
            logger.debug(f"Successfully cached result for key: {cache_key}")
            return True
        else:
            logger.warning("Redis client not available for caching")
            return False
    except Exception as e:
        logger.error(f"Error caching result for key {cache_key}: {str(e)}")
        return False

async def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Get cached result from Redis with enhanced error handling."""
    try:
        if redis_client:
            logger.debug(f"Retrieving cached result for key: {cache_key}")
            cached_data = redis_client.get(cache_key)
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Found cached result for key: {cache_key}")
                return result
            else:
                logger.debug(f"No cached result found for key: {cache_key}")
                return None
        else:
            logger.warning("Redis client not available for cache retrieval")
            return None
    except Exception as e:
        logger.error(f"Error retrieving cached result for key {cache_key}: {str(e)}")
        return None

# Fixed: Enhanced ReasoningTool with proper async support and round-robin API
class ReasoningTool:
    """Enhanced tool for applying reasoning to financial data using Gemini 1.5 Flash model"""
    
    @staticmethod
    async def reason_on_data(data: Union[str, Dict, List], query: str, 
                           max_words: int = 500, use_cache: bool = True) -> Dict[str, Any]:
        """Enhanced reasoning with proper async handling and retry mechanism"""
        logger.debug(f"Starting reasoning task - Query: {query[:100]}...")
        
        try:
            # Generate cache key
            cache_key = generate_cache_key("reasoning", query=query, data=str(data)[:100])
            
            # Check cache
            if use_cache:
                cached = await get_cached_result(cache_key)
                if cached:
                    logger.debug("Using cached reasoning result")
                    return cached
            
            # Format data for prompt
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False, indent=2)[:2000]
            else:
                data_str = str(data)[:2000]
            
            prompt = f"""Analyze this financial data and answer the query concisely (max {max_words} words).

Query: {query}

Data: {data_str}

Please provide:
1. Clear analysis based on the data
2. Key insights and recommendations
3. Confidence level (0.0 to 1.0)

Format your response with "Confidence: X.X" at the end."""

            # Use Gemini API with retry mechanism
            logger.debug("Calling Gemini API for reasoning")
            reasoning = await gemini_api.generate_content(prompt)
            
            # Extract confidence score
            confidence_match = re.search(r"confidence:\s*([0-9.]+)", reasoning.lower())
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            confidence = max(0.0, min(1.0, confidence))
            
            # Extract sources from data
            sources = []
            if isinstance(data, dict) and "sources" in data:
                sources.extend(data["sources"])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "source" in item:
                        sources.append(item["source"])
            
            result = {
                "reasoning": reasoning,
                "confidence": confidence,
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "cache_key": cache_key
            }
            
            # Cache result
            if use_cache:
                await cache_result(cache_key, result, ttl=1800)
            
            logger.debug(f"Reasoning completed successfully with confidence: {confidence}")
            return result
            
        except Exception as e:
            logger.error(f"ReasoningTool error: {str(e)}")
            return {
                "reasoning": f"Analysis failed due to technical issues: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Fixed: Enhanced VectorSearchTool with proper ChromaDB handling
class VectorSearchTool:
    """Enhanced Vector Search and RAG Tool with proper ChromaDB handling"""
    
    def __init__(self):
        """Initialize ChromaDB client and collection with proper error handling"""
        self.chroma_client = None
        self.chroma_collection = None
        logger.debug("Initializing VectorSearchTool")
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./chromadb_data",
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection("alphasage_chunks")
                logger.debug("Connected to existing ChromaDB collection: alphasage_chunks")
            except Exception:
                try:
                    self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name="alphasage_chunks",
                        metadata={"description": "Financial analysis chunks for AlphaSage"}
                    )
                    logger.debug("Created new ChromaDB collection: alphasage_chunks")
                except Exception as create_error:
                    logger.error(f"Failed to create ChromaDB collection: {create_error}")
                    raise
                    
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    async def search_knowledge_base(
        self, 
        query: str, 
        company_name: str = None,
        filters: Dict[str, Any] = None,
        max_results: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhanced vector search with proper error handling and caching"""
        logger.debug(f"Searching knowledge base - Query: {query[:50]}...")
        
        try:
            # Validate inputs
            if not query.strip():
                logger.warning("Empty query provided to vector search")
                return []
            
            # Generate cache key
            cache_key = None
            if use_cache:
                cache_key = generate_cache_key(
                    "vector_search", 
                    query=query, 
                    company=company_name or "all",
                    filters=str(filters) if filters else "none"
                )
                cached_result = await get_cached_result(cache_key)
                if cached_result:
                    logger.debug("Using cached vector search result")
                    return cached_result
            
            # Check if collection is available
            if not self.chroma_collection:
                logger.warning("ChromaDB collection not available")
                return []
            
            # Prepare filters
            where_clause = {}
            if company_name:
                where_clause["company_name"] = company_name
            if filters:
                where_clause.update(filters)
            
            # Perform search
            logger.debug(f"Executing ChromaDB query with filters: {where_clause}")
            search_params = {
                "query_texts": [query],
                "n_results": max_results
            }
            
            if where_clause:
                search_params["where"] = where_clause
            
            results = await asyncio.to_thread(self.chroma_collection.query, **search_params)
            
            # Format results
            formatted_results = []
            if results and "documents" in results and results["documents"]:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    try:
                        formatted_result = {
                            "content": text,
                            "metadata": metadata or {},
                            "similarity_score": max(0.0, 1.0 - (distance / 2.0)),
                            "relevance_score": max(0.0, 1.0 - (distance / 2.0)),
                            "source": metadata.get("source", "Unknown") if metadata else "Unknown",
                            "company_name": metadata.get("company_name", "") if metadata else "",
                            "category": metadata.get("category", "") if metadata else "",
                            "rank": i + 1
                        }
                        formatted_results.append(formatted_result)
                    except Exception as format_error:
                        logger.warning(f"Error formatting search result {i}: {format_error}")
                        continue
            
            # Cache the result
            if cache_key and formatted_results:
                await cache_result(cache_key, formatted_results, ttl=1800)
            
            logger.debug(f"Vector search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"VectorSearchTool error: {str(e)}")
            return []

# Fixed: Enhanced YFinanceAgentTool with async ticker validation
class YFinanceAgentTool:
    """Enhanced tool for fetching financial data from Yahoo Finance"""
    
    @staticmethod
    async def fetch_financial_data(
        ticker: str, 
        metric: str, 
        period: str = "5y",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Enhanced financial data fetching with async ticker validation"""
        logger.debug(f"Fetching financial data - Ticker: {ticker}, Metric: {metric}")
        
        try:
            # Validate ticker asynchronously
            if not await YFinanceAgentTool._validate_ticker_async(ticker):
                logger.warning(f"Invalid ticker: {ticker}, attempting MongoDB fallback")
                return await YFinanceAgentTool._mongodb_fallback(ticker, metric)
            
            # Generate cache key
            cache_key = generate_cache_key("financial_data", ticker=ticker, metric=metric, period=period)
            
            # Check cache
            if use_cache:
                cached = await get_cached_result(cache_key)
                if cached:
                    logger.debug(f"Using cached financial data for {ticker}")
                    return cached
            
            # Fetch data from yfinance
            logger.debug(f"Fetching from yfinance - {ticker}/{metric}")
            stock = yf.Ticker(ticker)
            
            # Different methods based on metric type
            result = None
            if metric.lower() in ['price', 'close', 'open', 'high', 'low', 'volume']:
                data = await asyncio.to_thread(stock.history, period=period)
                if not data.empty:
                    metric_key = 'Close' if metric.lower() == 'price' else metric.title()
                    if metric_key not in data.columns:
                        metric_key = 'Close'
                    
                    result = {
                        "success": True,
                        "data": [{
                            "value": float(data[metric_key].iloc[-1]),
                            "date": data.index[-1].strftime("%Y-%m-%d"),
                            "metric": metric
                        }],
                        "history": data[metric_key].astype(float).tolist(),
                        "dates": data.index.strftime("%Y-%m-%d").tolist(),
                        "ticker": ticker,
                        "sources": ["yfinance"]
                    }
            else:
                # Get from info
                info = await asyncio.to_thread(lambda: stock.info)
                if info and metric.lower().replace(' ', '_') in info:
                    value = info[metric.lower().replace(' ', '_')]
                    result = {
                        "success": True,
                        "data": [{
                            "value": float(value) if value is not None else None,
                            "metric": metric
                        }],
                        "ticker": ticker,
                        "sources": ["yfinance_info"]
                    }
            
            if result:
                # Cache successful result
                await cache_result(cache_key, result, ttl=3600)
                logger.debug(f"Successfully fetched financial data for {ticker}")
                return result
            else:
                logger.warning(f"No data found for {ticker}/{metric}")
                return await YFinanceAgentTool._mongodb_fallback(ticker, metric)
            
        except Exception as e:
            logger.error(f"YFinanceAgentTool error for {ticker}/{metric}: {str(e)}")
            return await YFinanceAgentTool._mongodb_fallback(ticker, metric)
    
    @staticmethod
    async def _validate_ticker_async(ticker: str) -> bool:
        """Validate ticker asynchronously"""
        try:
            logger.debug(f"Validating ticker: {ticker}")
            stock = yf.Ticker(ticker)
            info = await asyncio.to_thread(lambda: stock.info)
            is_valid = bool(info and 'regularMarketPrice' in info)
            logger.debug(f"Ticker {ticker} validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.warning(f"Ticker validation error for {ticker}: {str(e)}")
            return False
    
    @staticmethod
    async def _mongodb_fallback(ticker: str, metric: str) -> Dict[str, Any]:
        """MongoDB fallback for financial data"""
        logger.debug(f"Attempting MongoDB fallback for {ticker}/{metric}")
        
        try:
            if not collection:
                return {"success": False, "error": "MongoDB not available"}
            
            # Search MongoDB for financial data
            query = {
                "ticker": ticker,
                "category": "Financial Metrics",
                "$or": [
                    {"content.metric": {"$regex": metric, "$options": "i"}},
                    {"content.text": {"$regex": metric, "$options": "i"}}
                ]
            }
            
            chunk = await asyncio.to_thread(collection.find_one, query)
            if chunk:
                content = chunk.get("content", {})
                logger.debug(f"Found MongoDB fallback data for {ticker}/{metric}")
                return {
                    "success": True,
                    "data": [{
                        "value": content.get("value"),
                        "metric": metric
                    }],
                    "ticker": ticker,
                    "sources": ["mongodb", str(chunk["_id"])]
                }
            else:
                logger.warning(f"No MongoDB fallback data found for {ticker}/{metric}")
                return {"success": False, "error": "No data found in MongoDB"}
                
        except Exception as e:
            logger.error(f"MongoDB fallback error: {str(e)}")
            return {"success": False, "error": str(e)}

# --------------------------------
# Enhanced Testing Functions
# --------------------------------

async def test_tool_comprehensive(tool_name: str, input_data: Dict[str, Any], expected: Any = None, test_name: str = "") -> Dict[str, Any]:
    """
    Enhanced comprehensive test function with better error handling
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
            frequency = input_data.get("frequency", "1d")
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
            # This method doesn't exist in the current code, so we'll skip it or create a placeholder
            logger.warning(f"Tool {tool_name} not implemented")
            result = {"error": "Tool not implemented"}
            
        else:
            error = f"Unknown tool: {tool_name}"
        
        # Measure memory after execution
        if memory_before:
            try:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
            except:
                memory_after = None
            
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
        "success": error is None and result is not None,
        "status": "PASS" if (error is None and result is not None) else "FAIL",
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
    """Enhanced test suite with better error handling"""
    print("\n" + "="*80)
    print("ALPHASAGE TOOLS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check system health first
    health = check_system_health()
    print(f"\n=== System Health Check ===")
    for service, status in health["dependencies"].items():
        status_text = "✓ ONLINE" if status == "connected" else "✗ OFFLINE"
        print(f"{service.upper():15}: {status_text}")
    
    print(f"\n=== Test Configuration ===")
    print(f"MongoDB Available: {health['dependencies'].get('mongodb') == 'connected'}")
    print(f"Redis Available: {health['dependencies'].get('redis') == 'connected'}")
    print(f"ChromaDB Available: {health['dependencies'].get('chromadb') == 'connected'}")
    print(f"Gemini API Available: {health['dependencies'].get('yfinance') == 'connected'}")
    
    all_test_results = []
    
    # 1. YFinanceNumberTool Tests
    print(f"\n{'='*60}")
    print("1. YFINANCE NUMBER TOOL TESTS")
    print(f"{'='*60}")
    
    yfinance_tests = [
        {
            "name": "basic_price_fetch",
            "input": {"ticker": "RELIANCE.NS", "metric": "price"},
            "description": "Basic stock price fetch"
        },
        {
            "name": "historical_data",
            "input": {"ticker": "TCS.NS", "metric": "Close", "period": "1y"},
            "description": "Historical price data"
        },
        {
            "name": "invalid_ticker",
            "input": {"ticker": "INVALIDTICKER", "metric": "price"},
            "description": "Invalid ticker handling"
        }
    ]
    
    for test_config in yfinance_tests:
        print(f"\nRunning: {test_config['description']}")
        try:
            result = await test_tool_comprehensive("financial_data", test_config["input"], test_name=test_config["name"])
            if result:
                all_test_results.append(result)
                print(f"Status: {result.get('status', 'N/A')} ({result.get('execution_time', 0):.3f}s)")
                if result.get('error'):
                    print(f"Error: {result['error']}")
            else:
                print("Test tool returned None")
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")

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
        if result:
            all_test_results.append(result)
            print(f"Status: {result.get('status', 'N/A')}")
        else:
            print("Test tool returned None or an error.")
        
    
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
    if health['dependencies'].get('yfinance') == 'connected':
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
    if health['dependencies'].get('chromadb') == 'connected':
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
    if health['dependencies'].get('mongodb') != 'connected':
        print("• Consider setting up MongoDB for enhanced functionality")
    if health['dependencies'].get('redis') != 'connected':
        print("• Consider setting up Redis for improved caching")
    if health['dependencies'].get('yfinance') != 'connected':
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
            serializable_result = result.copy()
            
            # Convert any non-serializable values to strings
            for key, value in serializable_result.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_result[key] = str(value)
                    
                # Handle nested non-serializable objects in output
                if key == 'output' and isinstance(value, (dict, list)):
                    serializable_result[key] = json.dumps(value, default=str)
            
            serializable_results.append(serializable_result)
        
        # Save the results to file
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

# Add validate_ticker function
def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol is valid and exists.
    
    Args:
        ticker: The ticker symbol to validate
        
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        # Basic format validation
        if not ticker or not isinstance(ticker, str):
            return False
            
        # Check if ticker exists in yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        return bool(info and 'regularMarketPrice' in info)
        
    except Exception as e:
        logger.error(f"Error validating ticker {ticker}: {str(e)}")
        return False

# Add get_company_name_from_ticker function
def get_company_name_from_ticker(ticker: str) -> str:
    """
    Get company name from ticker symbol using yfinance.
    
    Args:
        ticker: The ticker symbol (e.g., 'GANECOS.NS')
        
    Returns:
        str: Company name if found, empty string otherwise
    """
    try:
        # Check if ticker exists
        if not validate_ticker(ticker):
            return ""
            
        # Get company info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get the company name from different possible fields
        company_name = (
            info.get('longName') or 
            info.get('shortName') or 
            info.get('name') or 
            ""
        )
        
        return company_name
        
    except Exception as e:
        logger.error(f"Error getting company name for ticker {ticker}: {str(e)}")
        return ""

# Add ticker to company name mapping for known companies
TICKER_TO_COMPANY = {
    "GANECOS.NS": "Ganesha Ecosphere Limited",
    "TATAMOTORS.NS": "Tata Motors",
    "RELIANCE.NS": "Reliance Industries"
}

def get_ticker_for_company(company_name: str) -> str:
    """
    Get ticker symbol for company name.
    
    Args:
        company_name: The company name
        
    Returns:
        str: Ticker symbol if found, empty string otherwise
    """
    # First check our known mappings
    for ticker, name in TICKER_TO_COMPANY.items():
        if name.lower() == company_name.lower():
            return ticker
            
    # If not found in mappings, try to find it
    try:
        # Search for the company using yfinance
        search_results = yf.Ticker(company_name).info
        if search_results and 'symbol' in search_results:
            return search_results['symbol']
    except Exception as e:
        logger.error(f"Error searching for ticker for company {company_name}: {str(e)}")
        
    return ""

if __name__ == "__main__":
    asyncio.run(main())
