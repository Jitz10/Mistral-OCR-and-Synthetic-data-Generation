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
import yfinance as yf
import chromadb

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
except ImportError as e:
    print("ERROR: Missing database libraries. Please install with:")
    print("pip install redis pymongo")
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
        ReasoningTool, YFinanceAgentTool, YFinanceNewsTool, 
        ArithmeticCalculationTool, VectorSearchTool,
        check_system_health, cache_result, get_cached_result, generate_cache_key
    )
    from microagent import AlphaSageMicroAgents, AgentResult
except ImportError as e:
    print("ERROR: Could not import tools.py or microagent.py. Make sure they're in the same directory.")
    # Don't raise - let it continue with warnings
    print(f"Import warning: {e}")

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

class AgentResult:
    """Result from an agent's analysis."""
    
    def __init__(
        self,
        agent_name: str = None,
        data: Dict = None,
        sources: List[str] = None,
        execution_time: float = 0.0,
        success: bool = True,
        error: str = None,
        cache_key: str = None,
        metadata: Dict = None
    ):
        self.agent_name = agent_name
        self.data = data or {}
        self.sources = sources or []
        self.execution_time = execution_time
        self.success = success
        self.error = error
        self.cache_key = cache_key
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "agent_name": self.agent_name,
            "data": self.data,
            "sources": self.sources,
            "execution_time": self.execution_time,
            "success": self.success,
            "error": self.error,
            "cache_key": self.cache_key,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentResult':
        """Create result from dictionary."""
        return cls(
            agent_name=data.get("agent_name"),
            data=data.get("data", {}),
            sources=data.get("sources", []),
            execution_time=data.get("execution_time", 0.0),
            success=data.get("success", True),
            error=data.get("error"),
            cache_key=data.get("cache_key"),
            metadata=data.get("metadata", {})
        )

class MacroAgent:
    """Macro-level analysis agent for comprehensive company analysis."""
    
    def __init__(self, name: str, llm_config: Dict):
        """Initialize the macro agent."""
        self.name = name
        self.llm_config = llm_config
        self.mongo_client = None
        self.redis_client = None
        self.chroma_client = None
        self.gemini_api = None
        self.analysis_tools = None
        self.reasoning_tool = None
        
    async def execute(self, company_name: str) -> AgentResult:
        """Execute the macro analysis."""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = generate_cache_key(self.name, company_name)
            
            # Check cache
            cached_result = get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Perform analysis
            analysis_result = await self._perform_analysis(company_name)
            
            # Create result
            result = AgentResult(
                agent_name=self.name,
                data=analysis_result,
                sources=[],
                execution_time=time.time() - start_time,
                success=True,
                cache_key=cache_key
            )
            
            # Cache result
            cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in macro agent execution: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                data={},
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
    async def _perform_analysis(self, company_name: str) -> Dict:
        """Perform the actual analysis."""
        try:
            # Initialize tools if not already done
            if not self.analysis_tools:
                self.analysis_tools = AnalysisTools(
                    mongo_client=self.mongo_client,
                    redis_client=self.redis_client,
                    chroma_client=self.chroma_client
                )
            
            if not self.reasoning_tool:
                self.reasoning_tool = ReasoningTool(
                    gemini_api=self.gemini_api,
                    llm_config=self.llm_config
                )
            
            # Get company data
            company_data = await self.analysis_tools.get_company_data(company_name)
            
            # Perform analysis based on agent type
            if self.name == "business_analysis":
                return await self._analyze_business(company_data)
            elif self.name == "financial_analysis":
                return await self._analyze_financial(company_data)
            elif self.name == "market_analysis":
                return await self._analyze_market(company_data)
            elif self.name == "risk_analysis":
                return await self._analyze_risk(company_data)
            else:
                raise ValueError(f"Unknown agent type: {self.name}")
                
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise

class AlphaSageMacroAgents:
    """Enhanced macro-level analysis agents for comprehensive company analysis"""
    
    def __init__(self):
        """Initialize macro agents with enhanced error handling"""
        try:
            # Initialize micro agents first
            from microagent import AlphaSageMicroAgents
            self.micro_agents = AlphaSageMicroAgents()
            
            # Initialize connections
            self.redis_client = self._init_redis()
            self.mongo_client = self._init_mongodb()
            self.chroma_client = self._init_chromadb()
            
            # Load LLM configuration
            self.llm_config = self._load_llm_config()
            
            # Initialize agents
            self.agents = {}
            self._configure_macro_agents()
            
            # Initialize tools
            self.tools = {
                'yfinance': YFinanceAgentTool(),
                'news': YFinanceNewsTool(),
                'arithmetic': ArithmeticCalculationTool(),
                'rag': VectorSearchTool(),
                'reasoning': ReasoningTool()
            }
            
            logger.info("Macro agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing macro agents: {str(e)}")
            raise
    
    def _configure_macro_agents(self):
        """Configure all macro agents with proper error handling"""
        try:
            # Initialize business analysis agent
            self.agents['business'] = MacroAgent(
                name="BusinessAnalysisAgent",
                llm_config=self.llm_config
            )
            
            # Initialize sector analysis agent
            self.agents['sector'] = MacroAgent(
                name="SectorAnalysisAgent",
                llm_config=self.llm_config
            )
            
            # Initialize deep dive agent
            self.agents['deepdive'] = MacroAgent(
                name="DeepDiveAgent",
                llm_config=self.llm_config
            )
            
            # Initialize current affairs agent
            self.agents['current_affairs'] = MacroAgent(
                name="CurrentAffairsAgent",
                llm_config=self.llm_config
            )
            
            # Initialize predictions agent
            self.agents['predictions'] = MacroAgent(
                name="PredictionsAgent",
                llm_config=self.llm_config
            )
            
            # Initialize concall analysis agent
            self.agents['concall'] = MacroAgent(
                name="ConcallAnalysisAgent",
                llm_config=self.llm_config
            )
            
            # Initialize risk analysis agent
            self.agents['risks'] = MacroAgent(
                name="RiskAnalysisAgent",
                llm_config=self.llm_config
            )
            
            logger.info("All macro agents configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring macro agents: {str(e)}")
            raise

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

    def _init_chromadb(self) -> Optional[chromadb.Client]:
        """Initialize ChromaDB connection"""
        try:
            client = chromadb.Client()
            logger.info("ChromaDB connection established for macro agents")
            return client
        except Exception as e:
            logger.warning(f"ChromaDB connection failed: {e}")
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

    def _get_ticker_for_company(self, company_name: str) -> str:
        """Get ticker symbol for a company name using micro agent"""
        try:
            # Use TickerResolutionAgent from micro agents
            ticker_result = self.micro_agents.resolve_ticker(company_name)
            if ticker_result and hasattr(ticker_result, 'success') and ticker_result.success:
                return ticker_result.data.get('ticker', f"{company_name.upper()}.NS")
            
            # Fallback to basic logic
            clean_name = company_name.replace(" Limited", "").replace(" Ltd", "").replace(" Private", "").replace(" Pvt", "")
            words = clean_name.split()
            
            if len(words) == 1:
                ticker = words[0][:6].upper()
            elif len(words) == 2:
                ticker = (words[0][:4] + words[1][:2]).upper()
            else:
                ticker = "".join([word[0] for word in words[:6]]).upper()
            
            return f"{ticker}.NS"
            
        except Exception as e:
            logger.warning(f"Ticker resolution failed for {company_name}: {e}")
            return f"{company_name.upper().replace(' ', '')}.NS"

    async def _query_mongodb_fallback(self, query: str, company_name: str, category: str = None) -> List[Dict[str, Any]]:
        """Query MongoDB as fallback when micro agents fail"""
        try:
            if not self.mongo_client:
                return []
            
            db = self.mongo_client.alphasage
            collection = db.financial_data
            
            # Build query
            mongo_query = {"$text": {"$search": f"{company_name} {query}"}}
            if category:
                mongo_query["category"] = category
            
            # Execute query
            results = list(collection.find(mongo_query).limit(10))
            
            # Transform results
            fallback_data = []
            for doc in results:
                fallback_data.append({
                    "_id": str(doc.get("_id", "")),
                    "content": doc.get("content", {}),
                    "source": doc.get("source", "mongodb"),
                    "timestamp": doc.get("timestamp", datetime.now().isoformat())
                })
            
            logger.info(f"MongoDB fallback returned {len(fallback_data)} results for {company_name}")
            return fallback_data
            
        except Exception as e:
            logger.warning(f"MongoDB fallback failed: {e}")
            return []

    async def retry_micro_call(self, func: Callable, *args, max_attempts: int = 3, **kwargs) -> Any:
        """Enhanced retry mechanism with MongoDB fallback"""
        attempt = 0
        last_exception = None
        company_name = kwargs.get('company_name', args[0] if args else None)
        
        # Ensure max_attempts is an integer
        if isinstance(max_attempts, str):
            max_attempts = int(max_attempts)
        
        while attempt < max_attempts:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Check if result is valid
                if result and hasattr(result, 'success') and result.success:
                    return result
                elif result:
                    # If result exists but not successful, try MongoDB fallback
                    if company_name and self.mongo_client:
                        fallback_data = await self._query_mongodb_fallback(
                            func.__name__,
                            company_name,
                            kwargs.get('category')
                        )
                        if fallback_data:
                            # Create a new AgentResult with fallback data
                            return AgentResult(
                                agent_name=result.agent_name,
                                data=fallback_data,
                                sources=result.sources + ["mongodb_fallback"],
                                execution_time=result.execution_time,
                                success=True
                            )
                    return result  # Return even if not successful
                    
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt >= max_attempts:
                    logger.error(f"Micro agent call failed after {max_attempts} attempts: {str(e)}")
                    break
                
                wait_time = (2 ** attempt) + 0.1
                logger.warning(f"Micro agent call attempt {attempt} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
        
        # Try MongoDB fallback as last resort
        if company_name and self.mongo_client:
            try:
                fallback_data = await self._query_mongodb_fallback(
                    func.__name__,
                    company_name,
                    kwargs.get('category')
                )
                if fallback_data:
                    return AgentResult(
                        agent_name=func.__name__.replace('_', ' ').title() + "Agent",
                        data=fallback_data,
                        sources=["mongodb_fallback"],
                        execution_time=0.0,
                        success=True
                    )
            except Exception as fallback_error:
                logger.error(f"MongoDB fallback also failed: {str(fallback_error)}")
        
        # Create fallback AgentResult
        return AgentResult(
            agent_name=func.__name__.replace('_', ' ').title() + "Agent",
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
        """Enhanced business analysis with proper micro agent integration"""
        start_time = time.time()
        cache_key = generate_cache_key("business_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="BusinessAnalysisAgent",
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
            
            # Fixed: Use the correct method name
            overview_result = await self.retry_micro_call(
                self.micro_agents.get_company_overview,
                company_name
            )
            if overview_result and overview_result.success:
                micro_agents_used.append("CompanyOverviewAgent")
                sources.extend(overview_result.sources)
            
            # Use BusinessModelAgent
            business_model_result = await self.retry_micro_call(
                self.micro_agents.analyze_business_model,
                company_name=company_name
            )
            if business_model_result.success:
                micro_agents_used.append("BusinessModelAgent")
                sources.extend(business_model_result.sources)
            
            # Use YFinanceNumberTool for financial metrics
            financial_data = await YFinanceAgentTool.fetch_financial_data(
                ticker,
                "market cap revenue profit"
            )
            if financial_data and financial_data.get('success'):
                tools_used.append("YFinanceNumberTool")
                sources.extend(financial_data.get('sources', []))
            
            # Use VectorSearchRAGTool for business insights
            business_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{company_name} business model strategy competitive advantage",
                company_name=company_name,
                filters={"category": "Business Analysis"},
                max_results=5
            )
            if business_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in business_chunks])
            
            # MongoDB fallback if insufficient data
            if not business_chunks or len(business_chunks) < 2:
                fallback_data = await self._query_mongodb_fallback(
                    "business model strategy competitive", 
                    company_name, 
                    "Business Analysis"
                )
                if fallback_data:
                    business_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for synthesis
            reasoning_data = {
                "overview": overview_result.data if overview_result.success else {},
                "business_model": business_model_result.data if business_model_result.success else {},
                "financial": financial_data.get('data', {}) if financial_data else {},
                "chunks_count": len(business_chunks)
            }
            
            business_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the business model and competitive advantages of {company_name}. What are the key strengths and strategic positioning?"
            )
            if business_reasoning:
                tools_used.append("ReasoningTool")
            
            # Compile business analysis data
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "business_summary": overview_result.data.get('summary', '') if overview_result.success else '',
                "business_model": business_model_result.data.get('model', '') if business_model_result.success else '',
                "sector": overview_result.data.get('sector', '') if overview_result.success else '',
                "industry": overview_result.data.get('industry', '') if overview_result.success else '',
                "market_cap": financial_data.get('data', {}).get('market_cap', 0) if financial_data else 0,
                "competitive_advantages": business_model_result.data.get('advantages', []) if business_model_result.success else [],
                "analysis": business_reasoning.get('reasoning', '') if business_reasoning else '',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="BusinessAnalysisAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            logger.error(f"Business analysis failed for {company_name}: {str(e)}")
            return MacroAgentResult(
                agent_name="BusinessAnalysisAgent",
                data={
                    "company_name": company_name,
                    "analysis_type": "business",
                    "error": str(e),
                    "macro_analysis": {},  # Empty fallback
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def analyze_sector(self, company_name: str, sector: str = None) -> MacroAgentResult:
        """Enhanced sector analysis with proper micro agent integration"""
        start_time = time.time()
        cache_key = generate_cache_key("sector_analysis", company_name=company_name, sector=sector)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="SectorAnalysisAgent",
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
            
            # Use SectorAnalysisAgent
            sector_result = await self.retry_micro_call(
                self.micro_agents.analyze_sector,
                sector or "Unknown",
                company_name
            )
            if sector_result.success:
                micro_agents_used.append("SectorAnalysisAgent")
                sources.extend(sector_result.sources)
            
            # Use CompetitorAnalysisAgent
            competitor_result = await self.retry_micro_call(
                self.micro_agents.analyze_competitors,
                company_name,
                sector or "Unknown"
            )
            if competitor_result.success:
                micro_agents_used.append("CompetitorAnalysisAgent")
                sources.extend(competitor_result.sources)
            
            # Use VectorSearchRAGTool for sector data
            sector_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{sector or company_name} sector industry trends analysis competitors",
                company_name=company_name,
                filters={"category": "Sector Analysis"},
                max_results=5
            )
            if sector_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in sector_chunks])
            
            # MongoDB fallback
            if not sector_chunks or len(sector_chunks) < 2:
                fallback_data = await self._query_mongodb_fallback(
                    f"sector industry trends {sector or ''}", 
                    company_name, 
                    "Sector Analysis"
                )
                if fallback_data:
                    sector_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for sector analysis
            reasoning_data = {
                "sector_analysis": sector_result.data if sector_result.success else {},
                "competitor_analysis": competitor_result.data if competitor_result.success else {},
                "sector": sector,
                "company": company_name
            }
            
            sector_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the sector dynamics and competitive landscape for {company_name} in the {sector or 'relevant'} sector."
            )
            if sector_reasoning:
                tools_used.append("ReasoningTool")
            
            result_data = {
                "company_name": company_name,
                "sector": sector or sector_result.data.get('sector', 'Unknown') if sector_result.success else 'Unknown',
                "sector_trends": sector_result.data.get('trends', []) if sector_result.success else [],
                "competitors": competitor_result.data.get('competitors', []) if competitor_result.success else [],
                "market_position": competitor_result.data.get('position', 'Unknown') if competitor_result.success else 'Unknown',
                "analysis": sector_reasoning.get('reasoning', '') if sector_reasoning else '',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="SectorAnalysisAgent",
                data=result_data,
                sources=sources,
                execution_time=time.time() - start_time,
                success=True,
                micro_agents_used=micro_agents_used,
                tools_used=tools_used,
                cache_key=cache_key
            )
            
        except Exception as e:
            self.log_macro_error(e, {"company_name": company_name, "sector": sector})
            return MacroAgentResult(
                agent_name="SectorAnalysisAgent",
                data={
                    "company_name": company_name,
                    "sector": sector or "Unknown",
                    "error_message": "Sector analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def deep_dive_company(self, company_name: str) -> MacroAgentResult:
        """Enhanced deep dive analysis with proper micro agent integration"""
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
            
            ticker = self._get_ticker_for_company(company_name)
            
            # Use HistoricalDataAgent
            historical_result = await self.retry_micro_call(
                self.micro_agents.fetch_historical_data,
                ticker,
                company_name,
                years=10
            )
            if historical_result.success:
                micro_agents_used.append("HistoricalDataAgent")
                sources.extend(historical_result.sources)
            
            # Use ManagementAnalysisAgent
            management_result = await self.retry_micro_call(
                self.micro_agents.analyze_management,
                company_name
            )
            if management_result.success:
                micro_agents_used.append("ManagementAnalysisAgent")
                sources.extend(management_result.sources)
            
            # Use GovernanceAnalysisAgent
            governance_result = await self.retry_micro_call(
                self.micro_agents.analyze_governance,
                company_name
            )
            if governance_result.success:
                micro_agents_used.append("GovernanceAnalysisAgent")
                sources.extend(governance_result.sources)
            
            # Use VectorSearchRAGTool for comprehensive data
            company_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{company_name} history management governance milestones achievements",
                company_name=company_name,
                max_results=10
            )
            if company_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in company_chunks])
            
            # MongoDB fallback for comprehensive data
            if not company_chunks or len(company_chunks) < 5:
                fallback_data = await self._query_mongodb_fallback(
                    "history management governance milestones", 
                    company_name
                )
                if fallback_data:
                    company_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for comprehensive analysis
            reasoning_data = {
                "historical": historical_result.data if historical_result.success else {},
                "management": management_result.data if management_result.success else {},
                "governance": governance_result.data if governance_result.success else {},
                "data_sources": len(company_chunks)
            }
            
            deepdive_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Provide a comprehensive deep dive analysis of {company_name} including company history, management quality, governance structure, and strategic evolution."
            )
            if deepdive_reasoning:
                tools_used.append("ReasoningTool")
            
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "company_history": historical_result.data.get('history', '') if historical_result.success else '',
                "key_milestones": historical_result.data.get('milestones', []) if historical_result.success else [],
                "management_team": management_result.data.get('team', []) if management_result.success else [],
                "management_quality": management_result.data.get('quality_score', 'Unknown') if management_result.success else 'Unknown',
                "governance_score": governance_result.data.get('score', 'Unknown') if governance_result.success else 'Unknown',
                "governance_highlights": governance_result.data.get('highlights', []) if governance_result.success else [],
                "comprehensive_analysis": deepdive_reasoning.get('reasoning', '') if deepdive_reasoning else '',
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
                    "company_name": company_name,
                    "error_message": "Deep dive analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def analyze_financials(self, company_name: str) -> MacroAgentResult:
        """Enhanced financial analysis with proper micro agent integration"""
        start_time = time.time()
        cache_key = generate_cache_key("financial_analysis", company_name=company_name)
        
        # Check cache first
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return MacroAgentResult(
                agent_name="FinancialAnalysisAgent",
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
            
            # Use ProfitabilityRatiosAgent
            profitability_result = await self.retry_micro_call(
                self.micro_agents.calculate_profitability_ratios,
                ticker=ticker,
                company_name=company_name
            )
            if profitability_result.success:
                micro_agents_used.append("ProfitabilityRatiosAgent")
                sources.extend(profitability_result.sources)
            
            # Use LeverageRatiosAgent
            leverage_result = await self.retry_micro_call(
                self.micro_agents.calculate_leverage_ratios,
                ticker=ticker,
                company_name=company_name
            )
            if leverage_result.success:
                micro_agents_used.append("LeverageRatiosAgent")
                sources.extend(leverage_result.sources)
            
            # Use LiquidityRatiosAgent
            liquidity_result = await self.retry_micro_call(
                self.micro_agents.calculate_liquidity_ratios,
                company_name=company_name
            )
            if liquidity_result.success:
                micro_agents_used.append("LiquidityRatiosAgent")
                sources.extend(liquidity_result.sources)
            
            # Use EfficiencyRatiosAgent
            efficiency_result = await self.retry_micro_call(
                self.micro_agents.calculate_efficiency_ratios,
                company_name=company_name
            )
            if efficiency_result.success:
                micro_agents_used.append("EfficiencyRatiosAgent")
                sources.extend(efficiency_result.sources)
            
            # Use YFinanceNumberTool for additional financial data
            financial_data = await YFinanceAgentTool.fetch_financial_data(
                ticker,
                "revenue profit debt cash flow"
            )
            if financial_data and financial_data.get('success'):
                tools_used.append("YFinanceNumberTool")
                sources.extend(financial_data.get('sources', []))
            
            # Use ArithmeticCalculationTool for custom calculations
            if profitability_result.success and leverage_result.success:
                calc_data = {
                    "profit_margin": profitability_result.data.get('profit_margin', 0),
                    "debt_equity": leverage_result.data.get('debt_equity', 0)
                }
                
                financial_health = ArithmeticCalculationTool.calculate_metrics(
                    calc_data,
                    "financial_health = (profit_margin * 100) / (1 + debt_equity)"
                )
                if financial_health:
                    tools_used.append("ArithmeticCalculationTool")
            
            # Use ReasoningTool for financial analysis
            reasoning_data = {
                "profitability": profitability_result.data if profitability_result.success else {},
                "leverage": leverage_result.data if leverage_result.success else {},
                "liquidity": liquidity_result.data if liquidity_result.success else {},
                "efficiency": efficiency_result.data if efficiency_result.success else {},
                "market_data": financial_data.get('data', {}) if financial_data else {}
            }
            
            financial_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the financial health and performance of {company_name}. What are the key financial strengths and concerns?"
            )
            if financial_reasoning:
                tools_used.append("ReasoningTool")
            
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "profitability_ratios": profitability_result.data if profitability_result.success else {},
                "leverage_ratios": leverage_result.data if leverage_result.success else {},
                "liquidity_ratios": liquidity_result.data if liquidity_result.success else {},
                "efficiency_ratios": efficiency_result.data if efficiency_result.success else {},
                "financial_health_score": financial_health.get('financial_health', 'Unknown') if 'financial_health' in locals() else 'Unknown',
                "financial_analysis": financial_reasoning.get('reasoning', '') if financial_reasoning else '',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result(cache_key, result_data, ttl=3600)  # 1 hour
            
            return MacroAgentResult(
                agent_name="FinancialAnalysisAgent",
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
                agent_name="FinancialAnalysisAgent",
                data={
                    "company_name": company_name,
                    "error_message": "Financial analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def analyze_current_affairs(self, company_name: str) -> MacroAgentResult:
        """Enhanced current affairs analysis with proper micro agent integration"""
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
            
            ticker = self._get_ticker_for_company(company_name)
            
            # Use NewsAnalysisAgent
            news_result = await self.retry_micro_call(
                self.micro_agents.analyze_news,
                ticker=ticker,
                company_name=company_name
            )
            if news_result.success:
                micro_agents_used.append("NewsAnalysisAgent")
                sources.extend(news_result.sources)
            
            # Use SentimentAnalysisAgent
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name=company_name,
                text="recent developments current events market sentiment news"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Use YFinanceNewsTool
            recent_news = await YFinanceNewsTool.fetch_company_news(
                ticker,
                max_results=20
            )
            if recent_news:
                tools_used.append("YFinanceNewsTool")
                sources.append("yfinance_news")
            
            # Use VectorSearchRAGTool for recent developments
            current_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{company_name} recent news developments events announcements current",
                company_name=company_name,
                max_results=10
            )
            if current_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in current_chunks])
            
            # MongoDB fallback for recent data
            if not current_chunks or len(current_chunks) < 3:
                fallback_data = await self._query_mongodb_fallback(
                    "recent news developments events announcements", 
                    company_name
                )
                if fallback_data:
                    current_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for event analysis
            reasoning_data = {
                "news_analysis": news_result.data if news_result.success else {},
                "sentiment": sentiment_result.data if sentiment_result.success else {},
                "news_count": len(recent_news) if recent_news else 0,
                "chunks_count": len(current_chunks)
            }
            
            affairs_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze recent current affairs and developments for {company_name}. What are the key events affecting the company?"
            )
            if affairs_reasoning:
                tools_used.append("ReasoningTool")
            
            # Process and categorize events
            events = []
            if recent_news:
                for article in recent_news[:10]:
                    events.append({
                        "title": article.get('title', 'Unknown event'),
                        "date": article.get('date', datetime.now().strftime('%Y-%m-%d')),
                        "summary": article.get('summary', 'No summary available')[:200] + "...",
                        "sentiment": self._analyze_news_sentiment(article.get('title', '') + article.get('summary', '')),
                        "source": "yfinance"
                    })
            
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "recent_events": events,
                "total_events": len(events),
                "overall_sentiment": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "news_analysis": news_result.data.get('analysis', '') if news_result.success else '',
                "current_affairs_summary": affairs_reasoning.get('reasoning', '') if affairs_reasoning else '',
                "sources": list(set(sources)),
                "micro_agents_used": micro_agents_used,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for shorter time due to time-sensitive nature
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
                    "company_name": company_name,
                    "error_message": "Current affairs analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    def _analyze_news_sentiment(self, text: str) -> str:
        """Enhanced sentiment analysis for news text"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['growth', 'profit', 'gain', 'increase', 'rise', 'up', 'positive', 'strong', 'good', 'excellent', 'beat', 'outperform', 'success', 'achieve', 'milestone', 'breakthrough']
        # Negative keywords  
        negative_words = ['loss', 'decline', 'fall', 'down', 'negative', 'weak', 'poor', 'miss', 'underperform', 'concern', 'risk', 'challenge', 'crisis', 'problem', 'issue']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count and positive_count > 0:
            return "positive"
        elif negative_count > positive_count and negative_count > 0:
            return "negative"
        else:
            return "neutral"

    async def predict_future(self, company_name: str, years: int = 3) -> MacroAgentResult:
        """Enhanced future predictions with proper micro agent integration"""
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
            historical_result = await self.retry_micro_call(
                self.micro_agents.fetch_historical_data,
                ticker=ticker,
                company_name=company_name,
                years=5
            )
            if historical_result.success:
                micro_agents_used.append("HistoricalDataAgent")
                sources.extend(historical_result.sources)
            
            # Use ScenarioAnalysisAgent for projections
            scenario_result = await self.retry_micro_call(
                self.micro_agents.perform_scenario_analysis,
                company_name=company_name,
                scenarios=["Optimistic", "Base Case", "Pessimistic"]
            )
            if scenario_result.success:
                micro_agents_used.append("ScenarioAnalysisAgent")
                sources.extend(scenario_result.sources)
            
            # Use GuidanceExtractionAgent for management projections
            guidance_result = await self.retry_micro_call(
                self.micro_agents.extract_guidance,
                company_name=company_name
            )
            if guidance_result.success:
                micro_agents_used.append("GuidanceExtractionAgent")
                sources.extend(guidance_result.sources)
            
            # Use YFinanceNumberTool for current metrics
            current_metrics = await YFinanceAgentTool.fetch_financial_data(
                ticker,
                "revenue profit growth"
            )
            if current_metrics and current_metrics.get('success'):
                tools_used.append("YFinanceNumberTool")
                sources.extend(current_metrics.get('sources', []))
            
            # Generate projections using ArithmeticCalculationTool
            projections = []
            current_year = datetime.now().year
            
            # Get base metrics
            base_revenue = current_metrics.get('data', {}).get('revenue', 1000) / 1000000 if current_metrics else 100  # in crores
            growth_rate = historical_result.data.get('avg_growth_rate', 0.12) if historical_result.success else 0.12
            
            for i in range(1, years + 1):
                year = current_year + i
                
                # Calculate projections using ArithmeticCalculationTool
                calc_data = {
                    "base_revenue": base_revenue,
                    "growth_rate": growth_rate,
                    "year": i
                }
                
                projection_calc = ArithmeticCalculationTool.calculate_metrics(
                    calc_data,
                    "projected_revenue = base_revenue * ((1 + growth_rate) ** year)"
                )
                
                if projection_calc:
                    tools_used.append("ArithmeticCalculationTool")
                    projected_revenue = projection_calc.get('projected_revenue', base_revenue * (1 + growth_rate) ** i)
                else:
                    projected_revenue = base_revenue * (1 + growth_rate) ** i
                
                projections.append({
                    "year": str(year),
                    "revenue": round(projected_revenue, 2),
                    "growth_rate": f"{growth_rate*100:.1f}%"
                })
            
            # Use ReasoningTool for projection analysis
            reasoning_data = {
                "historical": historical_result.data if historical_result.success else {},
                "scenarios": scenario_result.data if scenario_result.success else {},
                "guidance": guidance_result.data if guidance_result.success else {},
                "projections": projections
            }
            
            prediction_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the financial projections for {company_name}. What are the key assumptions and potential risks to these forecasts?"
            )
            if prediction_reasoning:
                tools_used.append("ReasoningTool")
            
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "projections": projections,
                "base_assumptions": {
                    "growth_rate": f"{growth_rate*100:.1f}%",
                    "base_revenue": f"{base_revenue:.2f} Cr",
                    "projection_method": "Historical trend analysis"
                },
                "scenarios": scenario_result.data if scenario_result.success else {},
                "management_guidance": guidance_result.data.get('guidance', '') if guidance_result.success else '',
                "analysis": prediction_reasoning.get('reasoning', '') if prediction_reasoning else '',
                "confidence_level": scenario_result.data.get('confidence', 0.7) if scenario_result.success else 0.7,
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
                    "company_name": company_name,
                    "projections": [{"year": str(datetime.now().year + 1), "revenue": 100.0, "growth_rate": "10.0%"}],
                    "error_message": "Future predictions analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def analyze_concall(self, company_name: str) -> MacroAgentResult:
        """Enhanced concall analysis with proper micro agent integration"""
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
            
            # Use ConcallAnalysisAgent
            concall_result = await self.retry_micro_call(
                self.micro_agents.analyze_concall,
                company_name=company_name
            )
            if concall_result.success:
                micro_agents_used.append("ConcallAnalysisAgent")
                sources.extend(concall_result.sources)
            
            # Use GuidanceExtractionAgent
            guidance_result = await self.retry_micro_call(
                self.micro_agents.extract_guidance,
                company_name=company_name
            )
            if guidance_result.success:
                micro_agents_used.append("GuidanceExtractionAgent")
                sources.extend(guidance_result.sources)
            
            # Use SentimentAnalysisAgent for management tone
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name=company_name,
                text="earnings call management commentary investor questions guidance outlook"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Use VectorSearchRAGTool for concall transcripts
            concall_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{company_name} earnings call concall transcript management commentary investor questions",
                company_name=company_name,
                filters={"category": "Earnings Call"},
                max_results=8
            )
            if concall_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in concall_chunks])
            
            # MongoDB fallback for concall data
            if not concall_chunks or len(concall_chunks) < 3:
                fallback_data = await self._query_mongodb_fallback(
                    "earnings call concall transcript management", 
                    company_name,
                    "Earnings Call"
                )
                if fallback_data:
                    concall_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for concall insights
            reasoning_data = {
                "concall_analysis": concall_result.data if concall_result.success else {},
                "guidance": guidance_result.data if guidance_result.success else {},
                "sentiment": sentiment_result.data if sentiment_result.success else {},
                "transcript_chunks": len(concall_chunks)
            }
            
            concall_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the earnings call insights for {company_name}. What are the key management messages, guidance, and investor concerns?"
            )
            if concall_reasoning:
                tools_used.append("ReasoningTool")
            
            # Extract key insights from transcripts
            key_insights = []
            for chunk in concall_chunks[:5]:
                content = chunk.get('content', {})
                if content.get('text'):
                    text = content['text']
                    if any(keyword in text.lower() for keyword in ['outlook', 'guidance', 'strategy', 'target', 'growth', 'plan']):
                        key_insights.append(text[:200] + "...")
            
            result_data = {
                "company_name": company_name,
                "concall_summary": concall_result.data.get('summary', '') if concall_result.success else '',
                "key_insights": key_insights,
                "management_guidance": guidance_result.data.get('guidance', '') if guidance_result.success else '',
                "management_tone": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "investor_concerns": concall_result.data.get('concerns', []) if concall_result.success else [],
                "forward_outlook": guidance_result.data.get('outlook', '') if guidance_result.success else '',
                "analysis": concall_reasoning.get('reasoning', '') if concall_reasoning else '',
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
                    "company_name": company_name,
                    "error_message": "Concall analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def analyze_risks(self, company_name: str) -> MacroAgentResult:
        """Enhanced risk analysis with proper micro agent integration"""
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
            
            # Use RiskAssessmentAgent
            risk_result = await self.retry_micro_call(
                self.micro_agents.assess_risks,
                company_name=company_name
            )
            if risk_result.success:
                micro_agents_used.append("RiskAssessmentAgent")
                sources.extend(risk_result.sources)
            
            # Use NewsAnalysisAgent for risk-related news
            news_result = await self.retry_micro_call(
                self.micro_agents.analyze_news,
                ticker=ticker,
                company_name=company_name
            )
            if news_result.success:
                micro_agents_used.append("NewsAnalysisAgent")
                sources.extend(news_result.sources)
            
            # Use SentimentAnalysisAgent for risk sentiment
            sentiment_result = await self.retry_micro_call(
                self.micro_agents.analyze_sentiment,
                company_name=company_name,
                text="risks challenges threats regulatory operational financial market competition"
            )
            if sentiment_result.success:
                micro_agents_used.append("SentimentAnalysisAgent")
                sources.extend(sentiment_result.sources)
            
            # Use YFinanceNewsTool for recent risk-related news
            recent_news = await YFinanceNewsTool.fetch_company_news(
                ticker,
                max_results=15
            )
            if recent_news:
                tools_used.append("YFinanceNewsTool")
                sources.append("yfinance_news")
            
            # Use VectorSearchRAGTool for risk information
            risk_chunks = await VectorSearchTool.search_knowledge_base(
                query=f"{company_name} risks challenges threats regulatory compliance financial operational",
                company_name=company_name,
                filters={"category": "Risk Analysis"},
                max_results=8
            )
            if risk_chunks:
                tools_used.append("VectorSearchRAGTool")
                sources.extend([f"chunk_{chunk.get('_id', 'unknown')}" for chunk in risk_chunks])
            
            # MongoDB fallback for risk data
            if not risk_chunks or len(risk_chunks) < 3:
                fallback_data = await self._query_mongodb_fallback(
                    "risks challenges threats regulatory financial", 
                    company_name,
                    "Risk Analysis"
                )
                if fallback_data:
                    risk_chunks.extend(fallback_data)
                    sources.extend([f"mongodb_{item.get('_id', 'unknown')}" for item in fallback_data])
            
            # Use ReasoningTool for comprehensive risk analysis
            reasoning_data = {
                "risk_assessment": risk_result.data if risk_result.success else {},
                "news_sentiment": sentiment_result.data if sentiment_result.success else {},
                "news_count": len(recent_news) if recent_news else 0,
                "risk_data_sources": len(risk_chunks)
            }
            
            risk_reasoning = await ReasoningTool.reason_on_data(
                json.dumps(reasoning_data),
                f"Analyze the comprehensive risk profile for {company_name}. What are the key risk factors, their likelihood, and potential mitigation strategies?"
            )
            if risk_reasoning:
                tools_used.append("ReasoningTool")
            
            # Categorize and analyze risks
            risk_categories = {
                "Financial": [],
                "Operational": [],
                "Market": [],
                "Regulatory": [],
                "Strategic": []
            }
            
            # Extract risks from news
            if recent_news:
                for article in recent_news:
                    title = article.get('title', '').lower()
                    summary = article.get('summary', '').lower()
                    
                    # Categorize risks based on keywords
                    if any(word in title + summary for word in ['debt', 'loss', 'profit', 'cash', 'revenue']):
                        risk_categories["Financial"].append(article.get('title', 'Unknown risk'))
                    elif any(word in title + summary for word in ['operation', 'production', 'supply', 'employee']):
                        risk_categories["Operational"].append(article.get('title', 'Unknown risk'))
                    elif any(word in title + summary for word in ['market', 'competition', 'competitor', 'share']):
                        risk_categories["Market"].append(article.get('title', 'Unknown risk'))
                    elif any(word in title + summary for word in ['regulation', 'law', 'compliance', 'government']):
                        risk_categories["Regulatory"].append(article.get('title', 'Unknown risk'))
                    else:
                        risk_categories["Strategic"].append(article.get('title', 'Unknown risk'))
            
            # Calculate risk scores using ArithmeticCalculationTool
            risk_score_data = {
                "financial_risks": len(risk_categories["Financial"]),
                "operational_risks": len(risk_categories["Operational"]),
                "market_risks": len(risk_categories["Market"]),
                "regulatory_risks": len(risk_categories["Regulatory"]),
                "strategic_risks": len(risk_categories["Strategic"])
            }
            
            risk_score_calc = ArithmeticCalculationTool.calculate_metrics(
                risk_score_data,
                "overall_risk_score = (financial_risks * 0.3) + (operational_risks * 0.25) + (market_risks * 0.2) + (regulatory_risks * 0.15) + (strategic_risks * 0.1)"
            )
            if risk_score_calc:
                tools_used.append("ArithmeticCalculationTool")
                overall_risk_score = risk_score_calc.get('overall_risk_score', 0)
            else:
                overall_risk_score = sum(len(risks) for risks in risk_categories.values()) * 0.2
            
            result_data = {
                "company_name": company_name,
                "ticker": ticker,
                "risk_categories": risk_categories,
                "overall_risk_score": round(overall_risk_score, 2),
                "risk_assessment": risk_result.data.get('assessment', '') if risk_result.success else '',
                "top_risks": risk_result.data.get('top_risks', []) if risk_result.success else [],
                "risk_sentiment": sentiment_result.data.get('sentiment', 'Neutral') if sentiment_result.success else 'Neutral',
                "analysis": risk_reasoning.get('reasoning', '') if risk_reasoning else '',
                "mitigation_strategies": risk_result.data.get('mitigation', []) if risk_result.success else [],
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
                    "company_name": company_name,
                    "error_message": "Risk analysis in progress",
                    "timestamp": datetime.now().isoformat()
                },
                sources=[],
                execution_time=time.time() - start_time,
                success=False,
                micro_agents_used=[],
                tools_used=[],
                error=str(e)
            )

    async def comprehensive_analysis(self, company_name: str) -> Dict[str, MacroAgentResult]:
        """
        Perform comprehensive analysis using all macro agents
        Returns a dictionary with results from all analysis types
        """
        start_time = time.time()
        logger.info(f"Starting comprehensive analysis for {company_name}")
        
        try:
            # Run all analyses concurrently
            analysis_tasks = {
                "business": self.analyze_business(company_name),
                "sector": self.analyze_sector(company_name),
                "deepdive": self.deep_dive_company(company_name),
                "financials": self.analyze_financials(company_name),
                "current_affairs": self.analyze_current_affairs(company_name),
                "predictions": self.predict_future(company_name),
                "concall": self.analyze_concall(company_name),
                "risks": self.analyze_risks(company_name)
            }
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*analysis_tasks.values(), return_exceptions=True)
            
            # Map results back to analysis types
            comprehensive_results = {}
            for i, (analysis_type, task) in enumerate(analysis_tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Error in {analysis_type} analysis: {result}")
                    comprehensive_results[analysis_type] = MacroAgentResult(
                        agent_name=f"{analysis_type.title()}Agent",
                        data={"error": str(result)},
                        sources=[],
                        execution_time=0.0,
                        success=False,
                        micro_agents_used=[],
                        tools_used=[],
                        error=str(result)
                    )
                else:
                    comprehensive_results[analysis_type] = result
            
            # Log summary
            successful_analyses = sum(1 for result in comprehensive_results.values() if result.success)
            total_time = time.time() - start_time
            
            logger.info(f"Comprehensive analysis completed for {company_name}: "
                       f"{successful_analyses}/{len(analysis_tasks)} analyses successful, "
                       f"Total time: {total_time:.2f}s")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {company_name}: {e}")
            # Return empty results for all analyses
            return {
                analysis_type: MacroAgentResult(
                    agent_name=f"{analysis_type.title()}Agent",
                    data={"error": "Comprehensive analysis failed"},
                    sources=[],
                    execution_time=0.0,
                    success=False,
                    micro_agents_used=[],
                    tools_used=[],
                    error=str(e)
                )
                for analysis_type in ["business", "sector", "deepdive", "financials", 
                                    "current_affairs", "predictions", "concall", "risks"]
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all macro agents and their dependencies"""
        status = {
            "macro_agents": {
                "total": len(self.agents),
                "configured": list(self.agents.keys()),
                "status": "operational" if self.agents else "not_configured"
            },
            "micro_agents": {
                "status": "operational" if self.micro_agents else "not_configured",
                "initialized": hasattr(self, 'micro_agents') and self.micro_agents is not None
            },
            "databases": {
                "redis": "connected" if self.redis_client else "disconnected",
                "mongodb": "connected" if self.mongo_client else "disconnected", 
                "chromadb": "connected" if self.chroma_client else "disconnected"
            },
            "llm_config": {
                "configured": self.llm_config is not None,
                "model": self.llm_config.get('config_list', [{}])[0].get('model', 'unknown') if self.llm_config else 'unknown'
            },
            "system_health": check_system_health(),
            "timestamp": datetime.now().isoformat()
        }
        
        return status

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        try:
            health_status = {
                "overall_status": "healthy",
                "components": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Check Redis
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    health_status["components"]["redis"] = "healthy"
                except:
                    health_status["components"]["redis"] = "unhealthy"
                    health_status["overall_status"] = "degraded"
            else:
                health_status["components"]["redis"] = "not_configured"
            
            # Check MongoDB
            if self.mongo_client:
                try:
                    self.mongo_client.admin.command('ping')
                    health_status["components"]["mongodb"] = "healthy"
                except:
                    health_status["components"]["mongodb"] = "unhealthy"
                    health_status["overall_status"] = "degraded"
            else:
                health_status["components"]["mongodb"] = "not_configured"
            
            # Check ChromaDB
            if self.chroma_client:
                try:
                    # Simple test to check if ChromaDB is accessible
                    self.chroma_client.heartbeat()
                    health_status["components"]["chromadb"] = "healthy"
                except:
                    health_status["components"]["chromadb"] = "unhealthy"
                    health_status["overall_status"] = "degraded"
            else:
                health_status["components"]["chromadb"] = "not_configured"
            
            # Check micro agents
            if self.micro_agents:
                health_status["components"]["micro_agents"] = "healthy"
            else:
                health_status["components"]["micro_agents"] = "unhealthy"
                health_status["overall_status"] = "degraded"
            
            # Check LLM config
            if self.llm_config:
                health_status["components"]["llm_config"] = "healthy"
            else:
                health_status["components"]["llm_config"] = "unhealthy"
                health_status["overall_status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Factory function for creating macro agents
def create_macro_agents() -> AlphaSageMacroAgents:
    """Factory function to create and initialize macro agents"""
    try:
        return AlphaSageMacroAgents()
    except Exception as e:
        logger.error(f"Failed to create macro agents: {e}")
        raise

# Example usage and testing
if __name__ == "__main__":
    async def test_macro_agents():
        """Test macro agents functionality"""
        try:
            # Create macro agents
            macro_agents = create_macro_agents()
            
            # Test company name
            test_company = "Reliance Industries"
            
            # Test individual analyses
            print(f"Testing macro agents with {test_company}...")
            
            # Test business analysis
            business_result = await macro_agents.analyze_business(test_company)
            print(f"Business Analysis: {'Success' if business_result.success else 'Failed'}")
            
            # Test comprehensive analysis
            print("Running comprehensive analysis...")
            comprehensive_results = await macro_agents.comprehensive_analysis(test_company)
            
            # Print summary
            for analysis_type, result in comprehensive_results.items():
                print(f"{analysis_type.title()}: {'Success' if result.success else 'Failed'}")
            
            # Test health check
            health = await macro_agents.health_check()
            print(f"System Health: {health['overall_status']}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            logger.error(f"Test failed: {e}")
    
    # Run the test
    asyncio.run(test_macro_agents())
