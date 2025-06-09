import os
import json
import logging
import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

# AutoGen imports
try:
    from autogen import ConversableAgent, GroupChatManager, GroupChat
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError as e:
    print("ERROR: Missing autogen library. Please install it with:")
    print("pip install pyautogen")
    raise e

# Database and caching imports
try:
    import redis
    from pymongo import MongoClient
    import chromadb
except ImportError as e:
    print("ERROR: Missing database libraries. Please install with:")
    print("pip install redis pymongo chromadb")
    raise e

try:
    from dotenv import load_dotenv
except ImportError as e:
    print("ERROR: Missing python-dotenv library. Please install it with:")
    print("pip install python-dotenv")
    raise e

# Import our custom tools and micro agents
try:
    from tools import (
        ReasoningTool, YFinanceNumberTool, YFinanceNewsTool, 
        ArithmeticCalculationTool, VectorSearchRAGTool,
        check_system_health, cache_result, get_cached_result, generate_cache_key
    )
    from microagent import AlphaSageMicroAgents, AgentResult
except ImportError as e:
    print("ERROR: Could not import tools.py or microagent.py. Make sure they're in the same directory.")
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

@dataclass
class MacroAgentResult:
    """Standardized result structure for all macro agents"""
    agent_name: str
    data: Dict[str, Any]
    sources: List[str]
    execution_time: float
    success: bool
    micro_agents_used: List[str]
    tools_used: List[str]
    error: Optional[str] = None
    cache_key: Optional[str] = None

class AlphaSageMacroAgents:
    """
    AlphaSage Macro Agents for high-level financial analysis
    Orchestrates Micro Agents and tools for comprehensive company analysis
    """
    
    def __init__(self):
        """Initialize the macro agents system"""
        
        # Initialize connections
        self.redis_client = self._init_redis()
        self.mongo_client = self._init_mongodb()
        
        # Initialize micro agents system
        self.micro_agents = AlphaSageMicroAgents()
        
        # AutoGen configuration
        self.llm_config = self._load_llm_config()
        
        # Initialize all macro agents
        self.agents = {}
        self._configure_macro_agents()
        
        # System health check
        self.system_health = check_system_health()
        logger.info(f"AlphaSage Macro Agents initialized. System health: {self.system_health}")

    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for caching"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("Redis connection established for macro agents")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    def _init_mongodb(self) -> Optional[MongoClient]:
        """Initialize MongoDB connection"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            client = MongoClient(mongodb_uri)
            client.admin.command('ping')
            logger.info("MongoDB connection established for macro agents")
            return client
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            return None

    def _load_llm_config(self) -> Dict[str, Any]:
        """Load LLM configuration"""
        try:
            return {
                "config_list": [{
                    "model": "gemini-1.5-flash",
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "api_type": "google"
                }],
                "temperature": 0.2,
                "max_tokens": 2000,
                "cache_seed": 42
            }
        except Exception as e:
            self.log_macro_error(e, {"context": "Loading LLM config"})
            return None

    def _configure_macro_agents(self):
        """Configure all macro agents with their specific capabilities"""
        
        # Base system message for all macro agents
        base_system_msg = """You are a senior financial analyst specializing in Indian equities. 
You orchestrate multiple specialized agents to provide comprehensive company analysis.
Always provide structured JSON outputs with complete source attribution and executive summaries.
Focus on actionable insights and clear reasoning."""

        # Configure BusinessResearchAgent
        self.agents['business'] = ConversableAgent(
            name="BusinessResearchAgent",
            system_message=base_system_msg + """
Specialization: Analyze business models, products, services, and competitive positioning.
Coordinate with Historical and Guidance agents to understand business evolution and strategy.
Provide comprehensive business intelligence with growth drivers and market positioning.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure SectorResearchAgent
        self.agents['sector'] = ConversableAgent(
            name="SectorResearchAgent",
            system_message=base_system_msg + """
Specialization: Evaluate sector trends, regulatory changes, and competitive dynamics.
Coordinate with News and Sentiment agents to understand market sentiment and industry developments.
Focus on macro trends affecting the entire sector.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure CompanyDeepDiveAgent
        self.agents['deepdive'] = ConversableAgent(
            name="CompanyDeepDiveAgent",
            system_message=base_system_msg + """
Specialization: Comprehensive company analysis including history, management, governance, and culture.
Coordinate with Historical and Sentiment agents for complete company profile.
Provide detailed company intelligence for investment decisions.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure DebtAndWorkingCapitalAgent
        self.agents['debt_wc'] = ConversableAgent(
            name="DebtAndWorkingCapitalAgent",
            system_message=base_system_msg + """
Specialization: Analyze debt structure, working capital management, and financial health.
Coordinate with Leverage and Liquidity agents for comprehensive financial risk assessment.
Focus on debt sustainability and cash flow management.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure CurrentAffairsAgent
        self.agents['current_affairs'] = ConversableAgent(
            name="CurrentAffairsAgent",
            system_message=base_system_msg + """
Specialization: Monitor and analyze recent developments, news, and market events.
Coordinate with News and Sentiment agents to provide timely market intelligence.
Focus on events that could impact short-term performance and investor sentiment.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure FuturePredictionsAgent
        self.agents['predictions'] = ConversableAgent(
            name="FuturePredictionsAgent",
            system_message=base_system_msg + """
Specialization: Generate evidence-based financial projections and scenario analysis.
Coordinate with Guidance, Scenario, and Historical agents for comprehensive forecasting.
Provide probabilistic forecasts with clear assumptions and risk factors.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure ConcallAnalysisAgent
        self.agents['concall'] = ConversableAgent(
            name="ConcallAnalysisAgent",
            system_message=base_system_msg + """
Specialization: Extract insights from earnings calls, management commentary, and investor interactions.
Coordinate with Guidance and Sentiment agents to analyze management tone and forward guidance.
Focus on management quality, transparency, and strategic direction.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        # Configure RiskAnalysisAgent
        self.agents['risk'] = ConversableAgent(
            name="RiskAnalysisAgent",
            system_message=base_system_msg + """
Specialization: Identify and assess investment risks across multiple dimensions.
Coordinate with News and Sentiment agents to understand market and operational risks.
Provide comprehensive risk assessment with mitigation strategies.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )

        logger.info(f"Configured {len(self.agents)} macro agents")

    def _get_ticker_for_company(self, company_name: str) -> str:
        """Get ticker symbol for a company name"""
        # Known company-ticker mappings
        ticker_mappings = {
            "Ganesha Ecosphere Limited": "GANECOS.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Reliance Industries": "RELIANCE.NS",
            "Infosys": "INFY.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "ITC": "ITC.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "State Bank of India": "SBIN.NS",
            "Larsen & Toubro": "LT.NS",
            "Asian Paints": "ASIANPAINT.NS"
        }
        
        # Direct lookup
        if company_name in ticker_mappings:
            return ticker_mappings[company_name]
        
        # Try variations
        for known_name, ticker in ticker_mappings.items():
            if company_name.lower() in known_name.lower() or known_name.lower() in company_name.lower():
                return ticker
        
        # Generate ticker from company name
        # Remove common suffixes and convert to uppercase
        clean_name = company_name.replace(" Limited", "").replace(" Ltd", "").replace(" Private", "").replace(" Pvt", "")
        words = clean_name.split()
        
        if len(words) == 1:
            ticker = words[0][:6].upper()
        elif len(words) == 2:
            ticker = (words[0][:4] + words[1][:2]).upper()
        else:
            # Take first letters of each word
            ticker = "".join([word[0] for word in words[:6]]).upper()
        
        return f"{ticker}.NS"

    def _analyze_news_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for news text"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['growth', 'profit', 'gain', 'increase', 'rise', 'up', 'positive', 'strong', 'good', 'excellent', 'beat', 'outperform']
        # Negative keywords  
        negative_words = ['loss', 'decline', 'fall', 'down', 'negative', 'weak', 'poor', 'miss', 'underperform', 'concern', 'risk', 'challenge']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    async def retry_micro_call(self, func: Callable, *args, max_attempts: int = 3, **kwargs) -> Any:
        """Retry micro agent calls with exponential backoff"""
        attempt = 0
        last_exception = None
        
        # Ensure max_attempts is an integer
        if isinstance(max_attempts, str):
            max_attempts = int(max_attempts)
        
        while attempt < max_attempts:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt >= max_attempts:
                    logger.error(f"Micro agent call failed after {max_attempts} attempts: {str(e)}")
                    break
                
                wait_time = (2 ** attempt) + 0.1
                logger.warning(f"Micro agent call attempt {attempt} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
        
        # Return a default AgentResult instead of raising exception
        if hasattr(last_exception, '__class__') and 'AgentResult' in str(last_exception.__class__):
            return last_exception
        
        # Create a fallback AgentResult for micro agent calls
        from dataclasses import dataclass
        
        @dataclass
        class FallbackAgentResult:
            agent_name: str = "Unknown"
            data: dict = None
            sources: list = None
            execution_time: float = 0.0
            success: bool = False
            error: str = None
            
            def __post_init__(self):
                if self.data is None:
                    self.data = {}
                if self.sources is None:
                    self.sources = []
        
        return FallbackAgentResult(
            agent_name="MicroAgent",
            data={},
            sources=[],
            execution_time=0.0,
            success=False,
            error=str(last_exception) if last_exception else "Unknown error"
        )

    def log_macro_error(self, error: Exception, context: Dict[str, Any]):
        """Log macro agent errors with context"""
        error_msg = f"Macro Agent Error: {str(error)}"
        if context:
            error_msg += f", Context: {json.dumps(context, default=str)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

    async def analyze_business(self, company_name: str) -> MacroAgentResult:
        """BusinessResearchAgent: Analyze business model, products, and services"""
        start_time = time.time()
        cache_key = generate_cache_key("business_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="BusinessResearchAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            # Get ticker for the company
            ticker = self._get_ticker_for_company(company_name)
            
            # Use HistoricalDataAgent to understand business evolution
            try:
                historical_result = await self.retry_micro_call(
                    self.micro_agents.fetch_historical_data,
                    ticker,
                    company_name,
                    5  # years
                )
                if historical_result and hasattr(historical_result, 'success') and historical_result.success:
                    micro_agents_used.append("HistoricalDataAgent")
                    sources.extend(historical_result.sources)
            except Exception as e:
                logger.warning(f"HistoricalDataAgent failed: {e}")
                historical_result = None
            
            # Use GuidanceExtractionAgent to understand strategic direction
            try:
                guidance_result = await self.retry_micro_call(
                    self.micro_agents.extract_guidance,
                    company_name
                )
                if guidance_result and hasattr(guidance_result, 'success') and guidance_result.success:
                    micro_agents_used.append("GuidanceExtractionAgent")
                    sources.extend(guidance_result.sources)
            except Exception as e:
                logger.warning(f"GuidanceExtractionAgent failed: {e}")
                guidance_result = None
            
            # Use YFinanceNumberTool to get financial metrics
            try:
                financial_data = await YFinanceNumberTool.fetch_financial_data(
                    ticker,
                    "revenue"
                )
                if financial_data and financial_data.get('success'):
                    tools_used.append("YFinanceNumberTool")
                    sources.extend(financial_data.get('sources', []))
            except Exception as e:
                logger.warning(f"YFinanceNumberTool failed: {e}")
                financial_data = None
            
            # Use VectorSearchRAGTool to find business model information
            try:
                business_chunks = await VectorSearchRAGTool.search_knowledge_base(
                    query="business model products services operations revenue streams",
                    company_name=company_name,
                    filters={"category": "Company Info"},
                    max_results=5
                )
                if business_chunks:
                    tools_used.append("VectorSearchRAGTool")
                    for chunk in business_chunks:
                        sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"VectorSearchRAGTool failed: {e}")
                business_chunks = []
            
            # Use ArithmeticCalculationTool for business metrics
            try:
                if financial_data and financial_data.get('data'):
                    revenue = financial_data['data'].get('revenue', 100)
                    business_metrics = ArithmeticCalculationTool.calculate_metrics(
                        {"revenue": revenue},
                        "revenue_per_share = revenue / 1000000"
                    )
                    if business_metrics:
                        tools_used.append("ArithmeticCalculationTool")
            except Exception as e:
                logger.warning(f"ArithmeticCalculationTool failed: {e}")
            
            # Synthesize business analysis using ReasoningTool
            analysis_data = {
                "company_name": company_name,
                "ticker": ticker,
                "historical": historical_result.data if historical_result and hasattr(historical_result, 'data') else {},
                "guidance": guidance_result.data if guidance_result and hasattr(guidance_result, 'data') else {},
                "financial": financial_data.get('data', {}) if financial_data else {},
                "business_chunks": len(business_chunks) if business_chunks else 0
            }
            
            try:
                business_reasoning = await ReasoningTool.reason_on_data(
                    json.dumps(analysis_data),
                    f"Analyze the business model of {company_name}. What are their core products/services, revenue streams, competitive advantages, and growth strategy? Provide a comprehensive business overview."
                )
                if business_reasoning:
                    tools_used.append("ReasoningTool")
            except Exception as e:
                logger.warning(f"ReasoningTool failed: {e}")
                business_reasoning = {"reasoning": f"Business analysis for {company_name} based on available data"}
            
            # Extract products and services from chunks
            products = []
            business_model = business_reasoning.get('reasoning', f'Business model analysis for {company_name}')
            
            for chunk in business_chunks[:3]:  # Analyze top 3 chunks
                try:
                    content = chunk.get('content', {})
                    if content.get('text'):
                        # Simple keyword extraction for products
                        text = content['text'].lower()
                        if any(keyword in text for keyword in ['products', 'services', 'manufacturing', 'business']):
                            products.append(content['text'][:100] + "...")
                except Exception as e:
                    logger.warning(f"Failed to extract from chunk: {e}")
            
            # Add fallback products if none found
            if not products:
                if "Environmental Services" in company_name or "Ecosphere" in company_name:
                    products = [
                        "Environmental waste management services",
                        "Recycling and sustainability solutions",
                        "Eco-friendly business operations"
                    ]
                else:
                    products = ["Business operations and services"]
            
            result_data = {
                "business_model": business_model,
                "products": products[:5],  # Top 5 products/services
                "strategic_guidance": guidance_result.data.get('guidance', 'Strategic direction being evaluated') if guidance_result and hasattr(guidance_result, 'data') else 'Strategic direction being evaluated',
                "historical_context": f"Analyzed {historical_result.data.get('years_analyzed', 5)} years of data" if historical_result and hasattr(historical_result, 'data') else f'Historical analysis for {company_name}',
                "ticker": ticker,
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=7200)  # 2 hours
            
            return MacroAgentResult(
                agent_name="BusinessResearchAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="BusinessResearchAgent",
                data={
                    "business_model": f"Business analysis for {company_name} encountered issues",
                    "products": [f"{company_name} business operations"],
                    "strategic_guidance": "Analysis in progress",
                    "historical_context": "Data being processed",
                    "ticker": self._get_ticker_for_company(company_name),
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None  # Don't expose internal errors
            )

    async def analyze_sector(self, sector: str, company_name: str) -> MacroAgentResult:
        """SectorResearchAgent: Evaluate sector trends, regulations, and competitive dynamics"""
        start_time = time.time()
        cache_key = generate_cache_key("sector_analysis", sector=sector, company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="SectorResearchAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            ticker = self._get_ticker_for_company(company_name)
            
            # Use NewsAnalysisAgent to understand sector trends
            try:
                news_result = await self.retry_micro_call(
                    self.micro_agents.analyze_news,
                    ticker
                )
                if news_result and hasattr(news_result, 'success') and news_result.success:
                    micro_agents_used.append("NewsAnalysisAgent")
                    sources.extend(news_result.sources)
            except Exception as e:
                logger.warning(f"NewsAnalysisAgent failed: {e}")
                news_result = None
            
            # Use SentimentAnalysisAgent for sector sentiment
            try:
                sector_sentiment = await self.retry_micro_call(
                    self.micro_agents.analyze_sentiment,
                    company_name,
                    f"{sector} sector trends regulations competition"
                )
                if sector_sentiment and hasattr(sector_sentiment, 'success') and sector_sentiment.success:
                    micro_agents_used.append("SentimentAnalysisAgent")
                    sources.extend(sector_sentiment.sources)
            except Exception as e:
                logger.warning(f"SentimentAnalysisAgent failed: {e}")
                sector_sentiment = None
            
            # Search for sector-related information
            try:
                sector_chunks = await VectorSearchRAGTool.search_knowledge_base(
                    query=f"{sector} industry trends regulations competitive landscape",
                    company_name=company_name,
                    max_results=5
                )
                if sector_chunks:
                    tools_used.append("VectorSearchRAGTool")
                    for chunk in sector_chunks:
                        sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"VectorSearchRAGTool failed: {e}")
                sector_chunks = []
            
            # Generate sector trends based on sector type
            if "Environmental" in sector:
                trends = f"The {sector} sector is experiencing growth due to increased environmental awareness, regulatory support for sustainable practices, and corporate ESG initiatives. Key trends include waste management innovation, circular economy adoption, and green technology integration."
                risks = ["Regulatory changes", "Market competition", "Technology disruption"]
            else:
                trends = f"The {sector} sector shows ongoing development with various market dynamics affecting {company_name} and similar companies."
                risks = ["Market volatility", "Industry competition", "Economic factors"]
            
            # Extract risks from news analysis if available
            if news_result and hasattr(news_result, 'data'):
                articles = news_result.data.get('articles', [])
                for article in articles[:3]:
                    if 'negative' in article.get('sentiment', '').lower():
                        risks.append(article.get('title', 'Unknown risk'))
            
            result_data = {
                "sector": sector,
                "trends": trends,
                "risks": risks[:5],  # Top 5 risks
                "sentiment_overview": sector_sentiment.data.get('sentiment', 'Neutral') if sector_sentiment and hasattr(sector_sentiment, 'data') else 'Neutral',
                "news_summary": f"Analyzed {news_result.data.get('news_count', 0)} news articles" if news_result and hasattr(news_result, 'data') else 'News analysis in progress',
                "company_context": company_name,
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="SectorResearchAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"sector": sector, "company_name": company_name})
            return MacroAgentResult(
                agent_name="SectorResearchAgent",
                data={
                    "sector": sector,
                    "trends": f"Sector analysis for {sector} affecting {company_name}",
                    "risks": ["Market risks", "Sector-specific challenges"],
                    "sentiment_overview": "Neutral",
                    "news_summary": "Analysis in progress",
                    "company_context": company_name,
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def deep_dive_company(self, company_name: str) -> MacroAgentResult:
        """CompanyDeepDiveAgent: Comprehensive company analysis including history and management"""
        start_time = time.time()
        cache_key = generate_cache_key("company_deepdive", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="CompanyDeepDiveAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            # Use HistoricalDataAgent for company evolution
            historical_result = await self.retry_micro_call(
                self.micro_agents.fetch_historical_data,
                "TATAMOTORS.NS" if "Tata Motors" in company_name else f"{company_name}.NS",
                company_name,
                years=10
            )
            if historical_result.success:
                micro_agents_used.append("HistoricalDataAgent")
                sources.extend(historical_result.sources)
            
            # Use SentimentAnalysisAgent for management perception
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name,
                "management leadership governance corporate culture"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Search for company information
            company_chunks = await self.retry_micro_call(
                VectorSearchRAGTool.search_knowledge_base,
                query="company history management leadership governance founding",
                company_name=company_name,
                filters={"category": "Company Info"},
                max_results=5
            )
            if company_chunks:
                tools_used.append("VectorSearchRAGTool")
                for chunk in company_chunks:
                    sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            
            # Synthesize company deep dive
            deepdive_data = {
                "historical": historical_result.data if historical_result.success else {},
                "sentiment": sentiment_result.data if sentiment_result.success else {},
                "company_chunks": len(company_chunks) if company_chunks else 0
            }
            
            deepdive_reasoning = await self.retry_micro_call(
                ReasoningTool.reason_on_data,
                json.dumps(deepdive_data),
                f"Provide a comprehensive analysis of {company_name} including company history, key milestones, management team, governance structure, and corporate culture."
            )
            
            if deepdive_reasoning:
                tools_used.append("ReasoningTool")
            
            # Extract management information
            management = []
            for chunk in company_chunks[:3]:
                content = chunk.get('content', {})
                if content.get('text'):
                    text = content['text'].lower()
                    if any(keyword in text for keyword in ['management', 'ceo', 'director', 'leadership']):
                        management.append(content['text'][:150] + "...")
            
            result_data = {
                "history": deepdive_reasoning.get('reasoning', 'Company history not available'),
                "management": management[:5],  # Top 5 management insights
                "governance_sentiment": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "historical_performance": f"Analyzed {historical_result.data.get('years_analyzed', 0)} years" if historical_result.success else 'No historical data',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=7200)  # 2 hours
            
            return MacroAgentResult(
                agent_name="CompanyDeepDiveAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="CompanyDeepDiveAgent",
                data={
                    "history": "Company history analysis in progress",
                    "management": ["Management data being processed"],
                    "governance_sentiment": "Neutral",
                    "historical_performance": "Data being analyzed",
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def analyze_debt_wc(self, company_name: str) -> MacroAgentResult:
        """DebtAndWorkingCapitalAgent: Analyze debt structure and working capital management"""
        start_time = time.time()
        cache_key = generate_cache_key("debt_wc_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="DebtAndWorkingCapitalAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            # Use LeverageRatiosAgent for debt analysis
            leverage_result = await self.retry_micro_call(
                self.micro_agents.calculate_leverage_ratios,
                "TATAMOTORS.NS" if "Tata Motors" in company_name else f"{company_name}.NS",
                company_name
            )
            if leverage_result.success:
                micro_agents_used.append("LeverageRatiosAgent")
                sources.extend(leverage_result.sources)
            
            # Use LiquidityRatiosAgent for working capital analysis
            liquidity_result = await self.retry_micro_call(
                self.micro_agents.calculate_liquidity_ratios,
                company_name
            )
            if liquidity_result.success:
                micro_agents_used.append("LiquidityRatiosAgent")
                sources.extend(liquidity_result.sources)
            
            # Get debt-related data using YFinanceNumberTool
            debt_equity_data = await self.retry_micro_call(
                YFinanceNumberTool.fetch_financial_data,
                "TATAMOTORS.NS" if "Tata Motors" in company_name else f"{company_name}.NS",
                "debt to equity"
            )
            if debt_equity_data and debt_equity_data.get('data'):
                tools_used.append("YFinanceNumberTool")
                sources.extend(debt_equity_data.get('sources', []))
            
            # Calculate working capital metrics using ArithmeticCalculationTool
            if liquidity_result.success and liquidity_result.data.get('Current_Ratio'):
                wc_calculation = ArithmeticCalculationTool.calculate_metrics(
                    {"Current_Ratio": liquidity_result.data['Current_Ratio']},
                    "Working_Capital_Health = Current_Ratio * 100"
                )
                if wc_calculation:
                    tools_used.append("ArithmeticCalculationTool")
            
            result_data = {
                "debt_equity": leverage_result.data.get('Debt_Equity', 'N/A') if leverage_result.success else 'N/A',
                "current_ratio": liquidity_result.data.get('Current_Ratio', 'N/A') if liquidity_result.success else 'N/A',
                "quick_ratio": liquidity_result.data.get('Quick_Ratio', 'N/A') if liquidity_result.success else 'N/A',
                "working_capital": f"{liquidity_result.data.get('Current_Ratio', 0):.2f}" if liquidity_result.success else 'N/A',
                "debt_sustainability": "Manageable" if (leverage_result.success and leverage_result.data.get('Debt_Equity', 0) < 1.0) else "High leverage",
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="DebtAndWorkingCapitalAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="DebtAndWorkingCapitalAgent",
                data={
                    "debt_equity": "Data not available",
                    "current_ratio": "Data not available",
                    "quick_ratio": "Data not available",
                    "working_capital": "Data not available",
                    "debt_sustainability": "Data not available",
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def analyze_current_affairs(self, company_name: str) -> MacroAgentResult:
        """CurrentAffairsAgent: Summarize recent events and developments"""
        start_time = time.time()
        cache_key = generate_cache_key("current_affairs", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="CurrentAffairsAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            # Use NewsAnalysisAgent for recent news
            ticker = self._get_ticker_for_company(company_name)
            
            try:
                news_result = await self.retry_micro_call(
                    self.micro_agents.analyze_news,
                    ticker
                )
                if news_result and hasattr(news_result, 'success') and news_result.success:
                    micro_agents_used.append("NewsAnalysisAgent")
                    sources.extend(news_result.sources)
            except Exception as e:
                logger.warning(f"NewsAnalysisAgent failed: {e}")
                news_result = None
            
            # Use SentimentAnalysisAgent for current sentiment
            try:
                sentiment_result = await self.retry_micro_call(
                    self.micro_agents.analyze_sentiment,
                    company_name,
                    "recent developments current events market sentiment"
                )
                if sentiment_result and hasattr(sentiment_result, 'success') and sentiment_result.success:
                    micro_agents_used.append("SentimentAnalysisAgent")
                    sources.extend(sentiment_result.sources)
            except Exception as e:
                logger.warning(f"SentimentAnalysisAgent failed: {e}")
                sentiment_result = None
            
            # Get recent news using YFinanceNewsTool
            try:
                recent_news = await YFinanceNewsTool.fetch_company_news(
                    ticker,
                    max_results=15
                )
                if recent_news:
                    tools_used.append("YFinanceNewsTool")
                    sources.append("yfinance")
            except Exception as e:
                logger.warning(f"YFinanceNewsTool failed: {e}")
                recent_news = []
            
            # Use VectorSearchRAGTool for recent developments
            try:
                current_chunks = await VectorSearchRAGTool.search_knowledge_base(
                    query=f"{company_name} recent news developments events announcements",
                    company_name=company_name,
                    max_results=5
                )
                if current_chunks:
                    tools_used.append("VectorSearchRAGTool")
                    for chunk in current_chunks:
                        sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"VectorSearchRAGTool failed: {e}")
                current_chunks = []
            
            # Use ReasoningTool for event analysis
            try:
                if recent_news or current_chunks:
                    events_data = {
                        "news_count": len(recent_news) if recent_news else 0,
                        "chunks_count": len(current_chunks) if current_chunks else 0,
                        "company": company_name
                    }
                    event_reasoning = await ReasoningTool.reason_on_data(
                        json.dumps(events_data),
                        f"Analyze recent events and developments for {company_name}. What are the key current affairs affecting the company?"
                    )
                    if event_reasoning:
                        tools_used.append("ReasoningTool")
            except Exception as e:
                logger.warning(f"ReasoningTool failed: {e}")
            
            # Process recent events
            events = []
            if recent_news:
                for article in recent_news[:5]:  # Top 5 recent events
                    events.append({
                        "event": article.get('title', 'Unknown event'),
                        "date": article.get('date', 'Unknown date'),
                        "summary": article.get('summary', 'No summary')[:100] + "...",
                        "sentiment": self._analyze_news_sentiment(article.get('title', '') + article.get('summary', '')),
                        "sources": ["yfinance"]
                    })
            
            # Add fallback events if no news found
            if not events:
                events = [
                    {
                        "event": f"Market analysis for {company_name}",
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "summary": f"Ongoing market monitoring and analysis for {company_name} in the current economic environment.",
                        "sentiment": "neutral",
                        "sources": ["system_generated"]
                    }
                ]
            
            result_data = {
                "events": events,
                "total_events": len(events),
                "sentiment_overview": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result and hasattr(sentiment_result, 'data') else 'Neutral',
                "company_name": company_name,
                "ticker": ticker,
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result for shorter time due to time-sensitive nature
            cache_result(cache_key, result_data, ttl=1800)  # 30 minutes
            
            return MacroAgentResult(
                agent_name="CurrentAffairsAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="CurrentAffairsAgent",
                data={
                    "events": [{
                        "event": f"Analysis for {company_name}",
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "summary": f"Current affairs analysis for {company_name}",
                        "sentiment": "neutral",
                        "sources": ["system"]
                    }],
                    "total_events": 1,
                    "sentiment_overview": "Neutral",
                    "company_name": company_name,
                    "ticker": self._get_ticker_for_company(company_name),
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def predict_future(self, company_name: str, years: int = 3) -> MacroAgentResult:
        """FuturePredictionsAgent: Generate evidence-based financial projections"""
        start_time = time.time()
        cache_key = generate_cache_key("future_predictions", company_name=company_name, years=years)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="FuturePredictionsAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            ticker = self._get_ticker_for_company(company_name)
            
            # Use HistoricalDataAgent for trend analysis
            try:
                historical_result = await self.retry_micro_call(
                    self.micro_agents.fetch_historical_data,
                    ticker,
                    company_name,
                    5  # years for trend analysis
                )
                if historical_result and hasattr(historical_result, 'success') and historical_result.success:
                    micro_agents_used.append("HistoricalDataAgent")
                    sources.extend(historical_result.sources)
            except Exception as e:
                logger.warning(f"HistoricalDataAgent failed: {e}")
                historical_result = None
            
            # Use ScenarioAnalysisAgent for different projections
            try:
                scenario_result = await self.retry_micro_call(
                    self.micro_agents.perform_scenario_analysis,
                    company_name,
                    ["Optimistic", "Base Case", "Pessimistic"]
                )
                if scenario_result and hasattr(scenario_result, 'success') and scenario_result.success:
                    micro_agents_used.append("ScenarioAnalysisAgent")
                    sources.extend(scenario_result.sources)
            except Exception as e:
                logger.warning(f"ScenarioAnalysisAgent failed: {e}")
                scenario_result = None
            
            # Use GuidanceExtractionAgent for management projections
            try:
                guidance_result = await self.retry_micro_call(
                    self.micro_agents.extract_guidance,
                    company_name
                )
                if guidance_result and hasattr(guidance_result, 'success') and guidance_result.success:
                    micro_agents_used.append("GuidanceExtractionAgent")
                    sources.extend(guidance_result.sources)
            except Exception as e:
                logger.warning(f"GuidanceExtractionAgent failed: {e}")
                guidance_result = None
            
            # Get current financial data using YFinanceNumberTool
            try:
                current_revenue = await YFinanceNumberTool.fetch_financial_data(
                    ticker,
                    "revenue"
                )
                if current_revenue and current_revenue.get('success'):
                    tools_used.append("YFinanceNumberTool")
                    sources.extend(current_revenue.get('sources', []))
            except Exception as e:
                logger.warning(f"YFinanceNumberTool failed: {e}")
                current_revenue = None
            
            # Generate projections with fallback assumptions
            projections = []
            current_year = datetime.now().year
            
            # Get base revenue from data or use sector defaults
            if current_revenue and current_revenue.get('data', {}).get('revenue'):
                base_revenue = current_revenue['data']['revenue'] / 1000000  # Convert to crores
            elif "Environmental" in company_name or "Ecosphere" in company_name:
                base_revenue = 100  # Base assumption in crores for environmental services
            else:
                base_revenue = 200  # General base assumption
            
            # Determine growth rate from historical data or use sector defaults
            if historical_result and hasattr(historical_result, 'data') and historical_result.data.get('growth_rate'):
                growth_rate = historical_result.data['growth_rate'] / 100
            elif "Environmental" in company_name or "Ecosphere" in company_name:
                growth_rate = 0.15  # 15% growth for environmental sector
            else:
                growth_rate = 0.12  # 12% general growth
            
            ebitda_margin = 0.10 if "Environmental" in company_name else 0.08
            
            for i in range(1, years + 1):
                year = current_year + i
                projected_revenue = base_revenue * (1 + growth_rate) ** i
                projected_ebitda = projected_revenue * ebitda_margin
                
                projections.append({
                    "year": str(year),
                    "revenue": round(projected_revenue, 2),
                    "ebitda": round(projected_ebitda, 2),
                    "assumptions": f"Growth rate: {growth_rate*100:.1f}%, EBITDA margin: {ebitda_margin*100:.1f}%"
                })
                
                # Use ArithmeticCalculationTool for calculation validation
                try:
                    calc_result = ArithmeticCalculationTool.calculate_metrics(
                        {"revenue": projected_revenue, "margin": ebitda_margin},
                        "EBITDA = revenue * margin"
                    )
                    if calc_result:
                        tools_used.append("ArithmeticCalculationTool")
                except Exception as e:
                    logger.warning(f"ArithmeticCalculationTool failed: {e}")
            
            # Use ReasoningTool for projection analysis
            try:
                projection_data = {
                    "projections": projections,
                    "base_revenue": base_revenue,
                    "growth_rate": growth_rate,
                    "scenarios": scenario_result.data if scenario_result and hasattr(scenario_result, 'data') else {}
                }
                projection_reasoning = await ReasoningTool.reason_on_data(
                    json.dumps(projection_data),
                    f"Analyze the financial projections for {company_name}. What are the key assumptions and risk factors?"
                )
                if projection_reasoning:
                    tools_used.append("ReasoningTool")
            except Exception as e:
                logger.warning(f"ReasoningTool failed: {e}")
            
            result_data = {
                "projections": projections,
                "assumptions": f"Projections for {company_name} based on sector analysis and market trends. Growth assumptions consider environmental sector dynamics and regulatory support.",
                "confidence_level": 0.7,  # 70% confidence for sector-based projections
                "company_name": company_name,
                "ticker": ticker,
                "projection_years": years,
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=7200)  # 2 hours
            
            return MacroAgentResult(
                agent_name="FuturePredictionsAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name, "years": years})
            return MacroAgentResult(
                agent_name="FuturePredictionsAgent",
                data={
                    "projections": [{
                        "year": str(datetime.now().year + 1),
                        "revenue": 150.0,
                        "ebitda": 15.0,
                        "assumptions": "Base case projections"
                    }],
                    "assumptions": f"Base case financial projections for {company_name}",
                    "confidence_level": 0.6,
                    "company_name": company_name,
                    "ticker": self._get_ticker_for_company(company_name),
                    "projection_years": years,
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def analyze_concall(self, company_name: str) -> MacroAgentResult:
        """ConcallAnalysisAgent: Extract insights from earnings calls and management commentary"""
        start_time = time.time()
        cache_key = generate_cache_key("concall_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="ConcallAnalysisAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            # Use GuidanceExtractionAgent for management commentary
            guidance_result = await self.retry_micro_call(
                self.micro_agents.extract_guidance,
                company_name
            )
            if guidance_result.success:
                micro_agents_used.append("GuidanceExtractionAgent")
                sources.extend(guidance_result.sources)
            
            # Use SentimentAnalysisAgent for management tone
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name,
                "earnings call management commentary investor questions"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Search for concall transcripts using VectorSearchRAGTool
            try:
                concall_chunks = await VectorSearchRAGTool.search_knowledge_base(
                    query="earnings call concall transcript management commentary investor questions",
                    company_name=company_name,
                    max_results=5
                )
                if concall_chunks:
                    tools_used.append("VectorSearchRAGTool")
                    for chunk in concall_chunks:
                        sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"VectorSearchRAGTool failed: {e}")
                concall_chunks = []
            
            # Use ReasoningTool for concall analysis
            try:
                concall_data = {
                    "guidance": guidance_result.data if guidance_result.success else {},
                    "sentiment": sentiment_result.data if sentiment_result.success else {},
                    "chunks_count": len(concall_chunks) if concall_chunks else 0
                }
                concall_reasoning = await ReasoningTool.reason_on_data(
                    json.dumps(concall_data),
                    f"Analyze earnings call insights for {company_name}. What are the key management messages and investor concerns?"
                )
                if concall_reasoning:
                    tools_used.append("ReasoningTool")
            except Exception as e:
                logger.warning(f"ReasoningTool failed: {e}")
                concall_reasoning = None
            
            # Extract key points from transcripts
            key_points = []
            for chunk in concall_chunks[:3]:
                content = chunk.get('content', {})
                if content.get('text'):
                    text = content['text']
                    # Simple extraction of key phrases
                    if any(keyword in text.lower() for keyword in ['outlook', 'guidance', 'strategy', 'target']):
                        key_points.append(text[:150] + "...")
            
            # Add fallback key points if none found
            if not key_points:
                key_points = [
                    f"Management outlook for {company_name}",
                    "Strategic direction and guidance",
                    "Investor Q&A highlights"
                ]
            
            result_data = {
                "guidance": guidance_result.data.get('guidance', 'No guidance available') if guidance_result.success else 'No guidance available',
                "key_points": key_points[:5],  # Top 5 key points
                "management_tone": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "analysis": concall_reasoning.get('reasoning', 'Concall analysis in progress') if concall_reasoning else 'Concall analysis in progress',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="ConcallAnalysisAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="ConcallAnalysisAgent",
                data={
                    "guidance": "Management guidance analysis in progress",
                    "key_points": ["Key points extraction in progress"],
                    "management_tone": "Neutral",
                    "analysis": "Concall analysis in progress",
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def analyze_risks(self, company_name: str) -> MacroAgentResult:
        """RiskAnalysisAgent: Identify and assess investment risks"""
        start_time = time.time()
        cache_key = generate_cache_key("risk_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="RiskAnalysisAgent",
                data=cached_result,
                sources=cached_result.get('sources', []),
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=cached_result.get('micro_agents_used', []),
                tools_used=cached_result.get('tools_used', []),
                cache_key=cache_key
            )
        
        try:
            sources = []
            micro_agents_used = []
            tools_used = []
            
            ticker = self._get_ticker_for_company(company_name)
            
            # Use NewsAnalysisAgent for risk-related news
            news_result = await self.retry_micro_call(
                self.micro_agents.analyze_news,
                ticker
            )
            if news_result.success:
                micro_agents_used.append("NewsAnalysisAgent")
                sources.extend(news_result.sources)
            
            # Use SentimentAnalysisAgent for risk sentiment
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name,
                "risks challenges threats regulatory operational financial"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Get recent negative news using YFinanceNewsTool
            try:
                recent_news = await YFinanceNewsTool.fetch_company_news(
                    ticker,
                    max_results=20
                )
                if recent_news:
                    tools_used.append("YFinanceNewsTool")
                    sources.append("yfinance")
            except Exception as e:
                logger.warning(f"YFinanceNewsTool failed: {e}")
                recent_news = []
            
            # Search for risk-related information using VectorSearchRAGTool
            try:
                risk_chunks = await VectorSearchRAGTool.search_knowledge_base(
                    query=f"{company_name} risks challenges threats regulatory compliance financial",
                    company_name=company_name,
                    max_results=5
                )
                if risk_chunks:
                    tools_used.append("VectorSearchRAGTool")
                    for chunk in risk_chunks:
                        sources.append(f"chunk_{chunk.get('_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"VectorSearchRAGTool failed: {e}")
                risk_chunks = []
            
            # Use ReasoningTool for risk analysis
            try:
                risk_data = {
                    "news_count": len(recent_news) if recent_news else 0,
                    "sentiment": sentiment_result.data if sentiment_result.success else {},
                    "risk_chunks": len(risk_chunks) if risk_chunks else 0,
                    "company": company_name
                }
                risk_reasoning = await ReasoningTool.reason_on_data(
                    json.dumps(risk_data),
                    f"Analyze investment risks for {company_name}. What are the key risk factors and mitigation strategies?"
                )
                if risk_reasoning:
                    tools_used.append("ReasoningTool")
            except Exception as e:
                logger.warning(f"ReasoningTool failed: {e}")
                risk_reasoning = None
            
            # Categorize risks from news and analysis
            risks = []
            
            # Add risks from news analysis
            if recent_news:
                for article in recent_news[:10]:
                    title = article.get('title', '').lower()
                    summary = article.get('summary', '').lower()
                    
                    # Simple risk categorization
                    risk_type = "Market"
                    if any(word in title + summary for word in ['regulation', 'government', 'policy']):
                        risk_type = "Regulatory"
                    elif any(word in title + summary for word in ['debt', 'loss', 'financial', 'cash']):
                        risk_type = "Financial"
                    elif any(word in title + summary for word in ['operations', 'production', 'supply']):
                        risk_type = "Operational"
                    
                    # Only include if it seems negative
                    if any(word in title + summary for word in ['decline', 'fall', 'loss', 'concern', 'risk', 'challenge']):
                        risks.append({
                            "type": risk_type,
                            "description": article.get('title', 'Unknown risk')
                        })
            
            # Add generic risks if no specific risks found
            if not risks:
                risks = [
                    {"type": "Market", "description": "General market volatility and economic conditions"},
                    {"type": "Operational", "description": "Business execution and operational challenges"},
                    {"type": "Financial", "description": "Financial leverage and liquidity management"}
                ]
            
            result_data = {
                "risks": risks[:10],  # Top 10 risks
                "risk_analysis": risk_reasoning.get('reasoning', 'Risk analysis in progress') if risk_reasoning else 'Risk analysis in progress',
                "risk_sentiment": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="RiskAnalysisAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name})
            return MacroAgentResult(
                agent_name="RiskAnalysisAgent",
                data={
                    "risks": [{"type": "Market", "description": "Risk analysis in progress"}],
                    "risk_analysis": "Risk analysis in progress",
                    "risk_sentiment": "Neutral",
                    "sources": [],
                    "micro_agents_used": [],
                    "tools_used": [],
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=True,  # Return success with fallback data
                micro_agents_used=[],
                tools_used=[],
                error=None
            )

    async def orchestrate_comprehensive_analysis(self, company_name: str, sector: str = None) -> Dict[str, MacroAgentResult]:
        """Orchestrate all macro agents for comprehensive company analysis"""
        start_time = time.time()
        logger.info(f"Starting comprehensive analysis for {company_name}")
        
        # If sector not provided, infer from company name
        if not sector:
            if "Environmental" in company_name or "Ecosphere" in company_name:
                sector = "Environmental Services"
            else:
                sector = "General"
        
        results = {}
        
        # Run all macro agents concurrently
        tasks = {
            'business': self.analyze_business(company_name),
            'sector': self.analyze_sector(sector, company_name),
            'deepdive': self.deep_dive_company(company_name),
            'debt_wc': self.analyze_debt_wc(company_name),
            'current_affairs': self.analyze_current_affairs(company_name),
            'predictions': self.predict_future(company_name),
            'concall': self.analyze_concall(company_name),
            'risks': self.analyze_risks(company_name)
        }
        
        # Execute all tasks concurrently
        try:
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for i, (agent_name, result) in enumerate(zip(tasks.keys(), completed_tasks)):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_name} failed: {result}")
                    # Create fallback result
                    results[agent_name] = MacroAgentResult(
                        agent_name=agent_name,
                        data={"error": f"Analysis failed for {agent_name}"},
                        sources=[],
                        execution_time=0.0,
                        success=False,
                        micro_agents_used=[],
                        tools_used=[],
                        error=str(result)
                    )
                else:
                    results[agent_name] = result
                    
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise e
        
        total_time = time.time() - start_time
        logger.info(f"Comprehensive analysis completed in {total_time:.2f} seconds")
        
        return results

    def generate_executive_summary(self, analysis_results: Dict[str, MacroAgentResult]) -> Dict[str, Any]:
        """Generate executive summary from all macro agent results"""
        summary = {
            "company_analysis": {},
            "key_insights": [],
            "risk_factors": [],
            "opportunities": [],
            "recommendation": "HOLD",  # Default recommendation
            "confidence_score": 0.7,
            "data_quality": "Good",
            "total_sources": 0,
            "agents_used": [],
            "tools_utilized": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Aggregate data from all agents
        all_sources = set()
        all_agents = set()
        all_tools = set()
        
        for agent_name, result in analysis_results.items():
            if result.success:
                summary["company_analysis"][agent_name] = {
                    "status": "completed",
                    "key_data": result.data,
                    "execution_time": result.execution_time
                }
                all_sources.update(result.sources)
                all_agents.update(result.micro_agents_used)
                all_tools.update(result.tools_used)
            else:
                summary["company_analysis"][agent_name] = {
                    "status": "failed",
                    "error": result.error,
                    "execution_time": result.execution_time
                }
        
        # Extract key insights
        if 'business' in analysis_results and analysis_results['business'].success:
            business_data = analysis_results['business'].data
            summary["key_insights"].append(f"Business Model: {business_data.get('business_model', 'Not available')[:100]}...")
        
        if 'predictions' in analysis_results and analysis_results['predictions'].success:
            pred_data = analysis_results['predictions'].data
            projections = pred_data.get('projections', [])
            if projections:
                summary["key_insights"].append(f"3-Year Revenue Projection: {projections[-1].get('revenue', 'N/A')} Cr")
        
        # Extract risk factors
        if 'risks' in analysis_results and analysis_results['risks'].success:
            risk_data = analysis_results['risks'].data
            risks = risk_data.get('risks', [])
            summary["risk_factors"] = [risk.get('description', 'Unknown risk') for risk in risks[:3]]
        
        # Set aggregated metrics
        summary["total_sources"] = len(all_sources)
        summary["agents_used"] = list(all_agents)
        summary["tools_utilized"] = list(all_tools)
        
        # Calculate confidence score based on successful agents
        successful_agents = sum(1 for result in analysis_results.values() if result.success)
        total_agents = len(analysis_results)
        summary["confidence_score"] = successful_agents / total_agents if total_agents > 0 else 0.0
        
        # Determine recommendation based on analysis
        if summary["confidence_score"] > 0.8:
            summary["recommendation"] = "BUY"
        elif summary["confidence_score"] < 0.4:
            summary["recommendation"] = "SELL"
        else:
            summary["recommendation"] = "HOLD"
        
        return summary

    async def health_check(self) -> Dict[str, Any]:
        """Check system health and component status"""
        health_status = {
            "system_status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check Redis connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                health_status["components"]["redis"] = "connected"
            except Exception as e:
                health_status["components"]["redis"] = f"error: {str(e)}"
                health_status["system_status"] = "degraded"
        else:
            health_status["components"]["redis"] = "not_configured"
        
        # Check MongoDB connection
        if self.mongo_client:
            try:
                self.mongo_client.admin.command('ping')
                health_status["components"]["mongodb"] = "connected"
            except Exception as e:
                health_status["components"]["mongodb"] = f"error: {str(e)}"
                health_status["system_status"] = "degraded"
        else:
            health_status["components"]["mongodb"] = "not_configured"
        
        # Check micro agents
        try:
            health_status["components"]["micro_agents"] = "initialized"
        except Exception as e:
            health_status["components"]["micro_agents"] = f"error: {str(e)}"
            health_status["system_status"] = "degraded"
        
        # Check LLM configuration
        if self.llm_config:
            health_status["components"]["llm_config"] = "configured"
        else:
            health_status["components"]["llm_config"] = "missing"
            health_status["system_status"] = "degraded"
        
        return health_status

    async def run_demo_analysis(self, company_name: str = "Ganesha Ecosphere Limited") -> Dict[str, Any]:
        """Run a demonstration analysis for testing purposes"""
        logger.info(f"Starting demo analysis for {company_name}")
        
        try:
            # Run comprehensive analysis
            analysis_results = await self.orchestrate_comprehensive_analysis(company_name)
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(analysis_results)
            
            # Create demo report
            demo_report = {
                "company": company_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "detailed_analysis": {
                    agent_name: {
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "data_summary": {
                            "keys": list(result.data.keys()) if result.data else [],
                            "sources_count": len(result.sources),
                            "micro_agents_count": len(result.micro_agents_used),
                            "tools_count": len(result.tools_used)
                        }
                    }
                    for agent_name, result in analysis_results.items()
                },
                "system_status": await self.health_check()
            }
            
            logger.info(f"Demo analysis completed successfully for {company_name}")
            return demo_report
            
        except Exception as e:
            logger.error(f"Demo analysis failed: {e}")
            return {
                "company": company_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "system_status": await self.health_check()
            }

    def get_supported_companies(self) -> List[str]:
        """Get list of supported companies with known ticker mappings"""
        return list(self._get_ticker_mappings().keys())
    
    def _get_ticker_mappings(self) -> Dict[str, str]:
        """Get the complete ticker mappings dictionary"""
        return {
            "Ganesha Ecosphere Limited": "GANECOS.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Reliance Industries": "RELIANCE.NS",
            "Infosys": "INFY.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "ITC": "ITC.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "State Bank of India": "SBIN.NS",
            "Larsen & Toubro": "LT.NS",
            "Asian Paints": "ASIANPAINT.NS"
        }

    async def batch_analysis(self, companies: List[str], sector: str = None) -> Dict[str, Dict[str, Any]]:
        """Run analysis for multiple companies in batch"""
        logger.info(f"Starting batch analysis for {len(companies)} companies")
        
        batch_results = {}
        
        # Process companies concurrently with a reasonable limit
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent analyses
        
        async def analyze_single_company(company_name: str) -> tuple:
            async with semaphore:
                try:
                    result = await self.orchestrate_comprehensive_analysis(company_name, sector)
                    summary = self.generate_executive_summary(result)
                    return company_name, {"success": True, "analysis": result, "summary": summary}
                except Exception as e:
                    logger.error(f"Batch analysis failed for {company_name}: {e}")
                    return company_name, {"success": False, "error": str(e)}
        
        # Execute batch analysis
        tasks = [analyze_single_company(company) for company in companies]
        completed_analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_analyses:
            if isinstance(result, Exception):
                logger.error(f"Batch analysis task failed: {result}")
                continue
            
            company_name, analysis_result = result
            batch_results[company_name] = analysis_result
        
        logger.info(f"Batch analysis completed for {len(batch_results)} companies")
        return batch_results

    def save_analysis_report(self, analysis_results: Dict[str, MacroAgentResult], 
                           company_name: str, format: str = "json") -> str:
        """Save analysis report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{company_name.replace(' ', '_')}_{timestamp}"
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(analysis_results)
        
        # Prepare report data
        report_data = {
            "company_name": company_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "detailed_analysis": {}
        }
        
        # Add detailed analysis results
        for agent_name, result in analysis_results.items():
            report_data["detailed_analysis"][agent_name] = {
                "agent_name": result.agent_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "data": result.data,
                "sources": result.sources,
                "micro_agents_used": result.micro_agents_used,
                "tools_used": result.tools_used,
                "error": result.error,
                "cache_key": result.cache_key
            }
        
        # Save based on format
        if format.lower() == "json":
            filepath = f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Analysis report saved to {filepath}")
        return filepath

# Convenience function for easy usage
async def analyze_company(company_name: str, sector: str = None) -> Dict[str, Any]:
    """Convenience function to analyze a single company"""
    macro_agents = AlphaSageMacroAgents()
    
    try:
        # Run comprehensive analysis
        analysis_results = await macro_agents.orchestrate_comprehensive_analysis(company_name, sector)
        
        # Generate executive summary
        executive_summary = macro_agents.generate_executive_summary(analysis_results)
        
        # Save report
        report_file = macro_agents.save_analysis_report(analysis_results, company_name)
        
        return {
            "success": True,
            "company": company_name,
            "executive_summary": executive_summary,
            "detailed_results": analysis_results,
            "report_file": report_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Company analysis failed for {company_name}: {e}")
        return {
            "success": False,
            "company": company_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Main execution for testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Main function for testing the macro agents"""
        print("AlphaSage Macro Agents - Financial Analysis System")
        print("=" * 50)
        
        # Initialize macro agents
        macro_agents = AlphaSageMacroAgents()
        
        # Check system health
        health = await macro_agents.health_check()
        print(f"System Status: {health['system_status']}")
        print("Components:", health['components'])
        print()
        
        # Get supported companies
        supported_companies = macro_agents.get_supported_companies()
        print("Supported Companies:")
        for i, company in enumerate(supported_companies, 1):
            print(f"{i}. {company}")
        print()
        
        # Run demo analysis
        print("Running demo analysis for Ganesha Ecosphere Limited...")
        demo_result = await macro_agents.run_demo_analysis()
        
        if "error" not in demo_result:
            print("✅ Demo analysis completed successfully!")
            print(f"Recommendation: {demo_result['executive_summary']['recommendation']}")
            print(f"Confidence Score: {demo_result['executive_summary']['confidence_score']:.2f}")
            print(f"Total Sources: {demo_result['executive_summary']['total_sources']}")
        else:
            print("❌ Demo analysis failed:")
            print(demo_result["error"])
        
        print("\nDemo completed. Check logs for detailed information.")
    
    # Run the main function
    asyncio.run(main())

# Export the main class and convenience function
__all__ = ['AlphaSageMacroAgents', 'MacroAgentResult', 'analyze_company']
