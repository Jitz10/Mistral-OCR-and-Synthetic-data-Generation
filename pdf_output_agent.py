import os
import json
import logging
import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

# AutoGen imports
try:
    from autogen import ConversableAgent, GroupChatManager, GroupChat
except ImportError as e:
    print("ERROR: Missing autogen library. Please install it with:")
    print("pip install pyautogen")
    raise e

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, blue, green, red
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib import colors
except ImportError as e:
    print("ERROR: Missing reportlab library. Please install it with:")
    print("pip install reportlab")
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

# Import our custom modules
try:
    from tools import (
        ReasoningTool, YFinanceAgentTool, YFinanceNewsTool, 
        ArithmeticCalculationTool, VectorSearchTool,
        check_system_health, cache_result, get_cached_result, generate_cache_key
    )
    from microagent import AlphaSageMicroAgents, AgentResult
    from macroagent import AlphaSageMacroAgents, MacroAgentResult
except ImportError as e:
    print("ERROR: Could not import required modules. Make sure they're in the same directory.")
    raise e

# Load environment variables
from dotenv import load_dotenv
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
logger = logging.getLogger('alphasage.pdf_output')

@dataclass
class PDFSection:
    """Represents a section in the PDF report"""
    title: str
    content: str
    charts: List[Dict] = None
    tables: List[Dict] = None
    sources: List[str] = None

class AlphaSagePDFAgent:
    """Enhanced PDF generation agent for comprehensive financial reports"""
    
    def __init__(self):
        """Initialize PDF agent with enhanced error handling"""
        logger.debug("Initializing AlphaSagePDFAgent")
        
        try:
            # Initialize micro and macro agents
            self.micro_agents = AlphaSageMicroAgents()
            self.macro_agents = AlphaSageMacroAgents()
            
            # Initialize database connections
            self.mongo_client = self._init_mongodb()
            self.redis_client = self._init_redis()
            
            # Initialize reportlab styles
            self.styles = getSampleStyleSheet()
            self._configure_custom_styles()
            
            # Initialize tools
            self.tools = {
                'yfinance': YFinanceAgentTool(),
                'news': YFinanceNewsTool(),
                'vector_search': VectorSearchTool(),
                'reasoning': ReasoningTool()
            }
            
            logger.info("PDF agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PDF agent: {str(e)}")
            raise
    
    def _configure_custom_styles(self):
        """Configure custom styles for the PDF report"""
        logger.debug("Configuring custom PDF styles")
        
        try:
            # Custom title style
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=HexColor('#1f4e79'),
                alignment=TA_CENTER
            ))
            
            # Custom heading style
            self.styles.add(ParagraphStyle(
                name='CustomHeading',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=HexColor('#2f5f8f'),
                borderWidth=1,
                borderColor=HexColor('#2f5f8f'),
                borderPadding=6
            ))
            
            # Custom body style
            self.styles.add(ParagraphStyle(
                name='CustomBody',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
                leftIndent=0,
                rightIndent=0
            ))
            
            # Custom bullet style
            self.styles.add(ParagraphStyle(
                name='CustomBullet',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leftIndent=20,
                bulletIndent=10
            ))
            
            logger.debug("Custom PDF styles configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring PDF styles: {str(e)}")
            raise
    
    def _init_mongodb(self) -> Optional[MongoClient]:
        """Initialize MongoDB connection with retry logic"""
        try:
            mongo_client = MongoClient(
                os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=20000,
                socketTimeoutMS=20000,
                maxPoolSize=10
            )
            mongo_client.admin.command('ping')
            logger.info("MongoDB connection established for PDF agent")
            return mongo_client
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            return None
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection"""
        try:
            redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            redis_client.ping()
            logger.info("Redis connection established for PDF agent")
            return redis_client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    async def generate_comprehensive_report(self, company_name: str, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive PDF report for a company"""
        logger.info(f"Starting comprehensive report generation for {company_name}")
        start_time = time.time()
        
        try:
            # Generate output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_company_name = company_name.replace(" ", "_").replace("/", "_")
                output_path = f"./reports/{safe_company_name}_Report_{timestamp}.pdf"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Fixed: Orchestrate comprehensive analysis with proper initialization
            logger.debug("Orchestrating comprehensive analysis")
            analysis_data = await self._orchestrate_comprehensive_analysis(company_name)
            
            # Fixed: Validate analysis data to avoid 'coroutine' errors
            logger.debug("Validating analysis data")
            data_quality = await self._validate_analysis_data(analysis_data)
            
            # Fixed: Generate consolidated insights with proper error handling
            logger.debug("Generating consolidated insights")
            consolidated_insights = await self._generate_consolidated_insights(
                analysis_data.get('macro_analysis', {}), 
                analysis_data.get('micro_analysis', {})
            )
            
            # Generate PDF sections
            logger.debug("Creating PDF sections")
            sections = await self._create_pdf_sections(company_name, analysis_data, consolidated_insights)
            
            # Create PDF document
            logger.debug(f"Creating PDF document at {output_path}")
            pdf_result = await self._create_pdf_document(output_path, company_name, sections, data_quality)
            
            execution_time = time.time() - start_time
            
            result = {
                "success": True,
                "output_path": output_path,
                "company_name": company_name,
                "sections_count": len(sections),
                "data_quality": data_quality,
                "execution_time": execution_time,
                "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"PDF report generated successfully: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "company_name": company_name,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

    async def _orchestrate_comprehensive_analysis(self, company_name: str) -> Dict[str, Any]:
        """Fixed: Orchestrate comprehensive analysis with proper initialization"""
        logger.debug(f"Orchestrating comprehensive analysis for {company_name}")
        
        # Fixed: Initialize analysis dictionaries properly
        macro_analysis = {}
        micro_analysis = {}
        
        try:
            # Run macro agent analysis with error handling
            logger.debug("Running macro agent analysis")
            try:
                macro_results = await self.macro_agents.comprehensive_analysis(company_name)
                if macro_results:
                    macro_analysis = {
                        key: result.data if hasattr(result, 'data') else result
                        for key, result in macro_results.items()
                    }
                    logger.debug(f"Macro analysis completed with {len(macro_analysis)} sections")
                else:
                    logger.warning("Macro analysis returned empty results")
            except Exception as macro_error:
                logger.error(f"Macro analysis failed: {str(macro_error)}")
                macro_analysis = {"error": str(macro_error)}
            
            # Run micro agent analysis with error handling
            logger.debug("Running micro agent analysis")
            try:
                # Fixed: Use proper async calls to micro agents
                micro_tasks = {
                    "financial_metrics": self.micro_agents.analyze_financial_metrics(company_name),
                    "market_news": self.micro_agents.analyze_market_news(company_name),
                    "business_operations": self.micro_agents.analyze_business_operations(company_name),
                    "risk_factors": self.micro_agents.analyze_risk_factors(company_name),
                    "valuation": self.micro_agents.analyze_valuation(company_name),
                    "profitability": self.micro_agents.analyze_profitability(company_name),
                    "historical": self.micro_agents.analyze_historical(company_name),
                    "sentiment": self.micro_agents.analyze_sentiment(company_name)
                }
                
                # Execute micro agent tasks
                micro_results = {}
                for agent_name, task in micro_tasks.items():
                    try:
                        result = await task
                        micro_results[agent_name] = result.data if hasattr(result, 'data') else result
                        logger.debug(f"Micro agent {agent_name} completed successfully")
                    except Exception as micro_error:
                        logger.warning(f"Micro agent {agent_name} failed: {str(micro_error)}")
                        micro_results[agent_name] = {"error": str(micro_error)}
                
                micro_analysis = micro_results
                logger.debug(f"Micro analysis completed with {len(micro_analysis)} agents")
                
            except Exception as micro_error:
                logger.error(f"Micro analysis failed: {str(micro_error)}")
                micro_analysis = {"error": str(micro_error)}
            
            # Compile comprehensive analysis
            analysis_data = {
                "company_name": company_name,
                "macro_analysis": macro_analysis,
                "micro_analysis": micro_analysis,
                "timestamp": datetime.now().isoformat(),
                "sources": self._extract_sources(macro_analysis, micro_analysis)
            }
            
            logger.debug("Comprehensive analysis orchestration completed")
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis orchestration: {str(e)}")
            return {
                "company_name": company_name,
                "macro_analysis": macro_analysis,  # Include partial results
                "micro_analysis": micro_analysis,  # Include partial results
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_sources(self, macro_analysis: Dict, micro_analysis: Dict) -> List[str]:
        """Extract all sources from analysis results"""
        sources = set()
        
        try:
            # Extract from macro analysis
            for analysis in macro_analysis.values():
                if isinstance(analysis, dict) and 'sources' in analysis:
                    sources.update(analysis['sources'])
            
            # Extract from micro analysis
            for analysis in micro_analysis.values():
                if isinstance(analysis, dict) and 'sources' in analysis:
                    sources.update(analysis['sources'])
            
            return list(sources)
            
        except Exception as e:
            logger.warning(f"Error extracting sources: {str(e)}")
            return ["yfinance", "mongodb", "chromadb"]

    async def _validate_analysis_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed: Validate analysis data with proper async handling"""
        logger.debug("Validating analysis data quality")
        
        try:
            validation_result = {
                "overall_quality": 0.0,
                "macro_quality": 0.0,
                "micro_quality": 0.0,
                "data_completeness": 0.0,
                "source_diversity": 0.0,
                "issues": []
            }
            
            macro_analysis = analysis_data.get('macro_analysis', {})
            micro_analysis = analysis_data.get('micro_analysis', {})
            
            # Validate macro analysis
            if macro_analysis:
                macro_valid_count = 0
                macro_total_count = len(macro_analysis)
                
                for key, analysis in macro_analysis.items():
                    if isinstance(analysis, dict) and not analysis.get('error'):
                        macro_valid_count += 1
                    else:
                        validation_result["issues"].append(f"Macro analysis '{key}' has issues")
                
                validation_result["macro_quality"] = macro_valid_count / macro_total_count if macro_total_count > 0 else 0.0
            else:
                validation_result["issues"].append("No macro analysis data available")
            
            # Validate micro analysis
            if micro_analysis:
                micro_valid_count = 0
                micro_total_count = len(micro_analysis)
                
                for key, analysis in micro_analysis.items():
                    if isinstance(analysis, dict) and not analysis.get('error'):
                        micro_valid_count += 1
                    else:
                        validation_result["issues"].append(f"Micro analysis '{key}' has issues")
                
                validation_result["micro_quality"] = micro_valid_count / micro_total_count if micro_total_count > 0 else 0.0
            else:
                validation_result["issues"].append("No micro analysis data available")
            
            # Calculate overall quality
            validation_result["overall_quality"] = (
                validation_result["macro_quality"] * 0.6 + 
                validation_result["micro_quality"] * 0.4
            )
            
            # Validate data completeness
            total_sections = len(macro_analysis) + len(micro_analysis)
            completed_sections = sum(1 for analysis in list(macro_analysis.values()) + list(micro_analysis.values()) 
                                   if isinstance(analysis, dict) and not analysis.get('error'))
            
            validation_result["data_completeness"] = completed_sections / total_sections if total_sections > 0 else 0.0
            
            # Validate source diversity
            sources = analysis_data.get('sources', [])
            validation_result["source_diversity"] = min(1.0, len(sources) / 5.0)  # Expect at least 5 sources
            
            logger.debug(f"Data validation completed - Overall quality: {validation_result['overall_quality']:.2f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating analysis data: {str(e)}")
            return {
                "overall_quality": 0.0,
                "macro_quality": 0.0,
                "micro_quality": 0.0,
                "data_completeness": 0.0,
                "source_diversity": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "error": str(e)
            }

    async def _generate_consolidated_insights(self, macro_analysis: Dict, micro_analysis: Dict) -> Dict[str, Any]:
        """Fixed: Generate consolidated insights with proper macro_analysis handling"""
        logger.debug("Generating consolidated insights")
        
        try:
            # Fixed: Handle missing macro_analysis with fallbacks
            if not macro_analysis:
                logger.warning("macro_analysis is missing, using MongoDB fallback")
                macro_analysis = await self._get_mongodb_fallback_data()
            
            # Prepare data for reasoning
            consolidation_data = {
                "macro_insights": macro_analysis,
                "micro_insights": micro_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Use reasoning tool for consolidation
            consolidation_query = """
            Consolidate the macro and micro analysis into key executive insights:
            1. Investment thesis (2-3 sentences)
            2. Key strengths (top 3)
            3. Key risks (top 3)  
            4. Financial health score (1-10)
            5. Recommendation (Buy/Hold/Sell with rationale)
            """
            
            reasoning_result = await ReasoningTool.reason_on_data(
                data=consolidation_data,
                query=consolidation_query,
                max_words=300
            )
            
            # Extract structured insights
            insights = {
                "executive_summary": reasoning_result.get('reasoning', 'Analysis in progress...'),
                "confidence": reasoning_result.get('confidence', 0.5),
                "investment_thesis": self._extract_investment_thesis(reasoning_result.get('reasoning', '')),
                "key_strengths": self._extract_key_points(reasoning_result.get('reasoning', ''), 'strengths'),
                "key_risks": self._extract_key_points(reasoning_result.get('reasoning', ''), 'risks'),
                "financial_health_score": self._extract_financial_score(reasoning_result.get('reasoning', '')),
                "recommendation": self._extract_recommendation(reasoning_result.get('reasoning', '')),
                "sources": reasoning_result.get('sources', []),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug("Consolidated insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating consolidated insights: {str(e)}")
            return {
                "executive_summary": f"Analysis for the company is in progress. Technical details: {str(e)[:100]}",
                "confidence": 0.3,
                "investment_thesis": "Investment analysis pending due to data processing.",
                "key_strengths": ["Company operational", "Market presence", "Business continuity"],
                "key_risks": ["Market volatility", "Operational challenges", "Regulatory changes"],
                "financial_health_score": 5.0,
                "recommendation": "Hold - Analysis pending",
                "sources": ["system_fallback"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_mongodb_fallback_data(self) -> Dict[str, Any]:
        """Get fallback data from MongoDB when macro_analysis is missing"""
        try:
            if not self.mongo_client:
                return {}
            
            # Try to get some basic company data from MongoDB
            db = self.mongo_client['alphasage']
            collection = db['alphasage_chunks']
            
            # Search for any company data (fixed: use synchronous find)
            company_docs = collection.find(
                {"content.type": {"$exists": True}},
                {"content": 1, "category": 1, "source": 1}
            ).limit(10)
            
            fallback_data = {}
            for doc in company_docs:
                category = doc.get('category', 'general')
                if category not in fallback_data:
                    fallback_data[category] = []
                fallback_data[category].append(doc.get('content', {}))
            
            return fallback_data
            
        except Exception as e:
            logger.warning(f"MongoDB fallback failed: {str(e)}")
            return {}
    
    def _extract_investment_thesis(self, text: str) -> str:
        """Extract investment thesis from reasoning text"""
        try:
            # Look for investment thesis patterns
            import re
            patterns = [
                r'investment thesis[:\s]+(.*?)(?:\n|\.)',
                r'thesis[:\s]+(.*?)(?:\n|\.)',
                r'investment[:\s]+(.*?)(?:\n|\.)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    return match.group(1).strip().capitalize()
            
            # Fallback to first sentence
            sentences = text.split('.')
            return sentences[0].strip() if sentences else "Investment analysis in progress"
            
        except Exception:
            return "Investment analysis in progress"
    
    def _extract_key_points(self, text: str, point_type: str) -> List[str]:
        """Extract key strengths or risks from text"""
        try:
            import re
            
            if point_type.lower() == 'strengths':
                patterns = [r'strengths?[:\s]+(.*?)(?:risks?|$)', r'positive[:\s]+(.*?)(?:negative|$)']
                default_points = ["Operational stability", "Market position", "Financial structure"]
            else:  # risks
                patterns = [r'risks?[:\s]+(.*?)(?:strengths?|$)', r'challenges?[:\s]+(.*?)(?:opportunities|$)']
                default_points = ["Market volatility", "Competitive pressure", "Regulatory changes"]
            
            for pattern in patterns:
                match = re.search(pattern, text.lower(), re.DOTALL)
                if match:
                    points_text = match.group(1)
                    # Split by common delimiters
                    points = re.split(r'[,;.\n]', points_text)
                    clean_points = [p.strip().capitalize() for p in points if p.strip() and len(p.strip()) > 5]
                    return clean_points[:3] if clean_points else default_points
            
            return default_points
            
        except Exception:
            return ["Analysis pending", "Data processing", "Review in progress"]
    
    def _extract_financial_score(self, text: str) -> float:
        """Extract financial health score from text"""
        try:
            import re
            
            # Look for score patterns
            patterns = [
                r'score[:\s]+([0-9.]+)',
                r'health[:\s]+([0-9.]+)',
                r'rating[:\s]+([0-9.]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    score = float(match.group(1))
                    return max(1.0, min(10.0, score))
            
            # Default to middle score
            return 5.5
            
        except Exception:
            return 5.0
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract investment recommendation from text"""
        try:
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['buy', 'purchase', 'invest', 'strong buy']):
                return "Buy - Based on analysis"
            elif any(word in text_lower for word in ['sell', 'exit', 'avoid']):
                return "Sell - Based on analysis"
            elif any(word in text_lower for word in ['hold', 'maintain', 'neutral']):
                return "Hold - Based on analysis"
            else:
                return "Hold - Analysis pending"
                
        except Exception:
            return "Hold - Analysis pending"

    async def _create_pdf_sections(self, company_name: str, analysis_data: Dict, consolidated_insights: Dict) -> List[PDFSection]:
        """Create PDF sections from analysis data"""
        logger.debug("Creating PDF sections")
        
        try:
            sections = []
            
            # Executive Summary Section
            sections.append(PDFSection(
                title="Executive Summary",
                content=self._format_executive_summary(company_name, consolidated_insights),
                sources=consolidated_insights.get('sources', [])
            ))
            
            # Financial Metrics Section
            financial_data = analysis_data.get('micro_analysis', {}).get('financial_metrics', {})
            sections.append(PDFSection(
                title="Financial Metrics Analysis",
                content=self._format_financial_metrics(financial_data),
                tables=[self._create_financial_metrics_table(financial_data)],
                sources=financial_data.get('sources', [])
            ))
            
            # Business Analysis Section
            business_data = analysis_data.get('macro_analysis', {}).get('business', {})
            sections.append(PDFSection(
                title="Business & Strategy Analysis",
                content=self._format_business_analysis(business_data),
                sources=business_data.get('sources', [])
            ))
            
            # Market & News Analysis Section
            news_data = analysis_data.get('micro_analysis', {}).get('market_news', {})
            sections.append(PDFSection(
                title="Market News & Sentiment",
                content=self._format_news_analysis(news_data),
                tables=[self._create_news_summary_table(news_data)],
                sources=news_data.get('sources', [])
            ))
            
            # Risk Analysis Section
            risk_data = analysis_data.get('macro_analysis', {}).get('risks', {})
            sections.append(PDFSection(
                title="Risk Assessment",
                content=self._format_risk_analysis(risk_data, consolidated_insights),
                sources=risk_data.get('sources', [])
            ))
            
            # Future Projections Section
            predictions_data = analysis_data.get('macro_analysis', {}).get('predictions', {})
            sections.append(PDFSection(
                title="Future Projections",
                content=self._format_projections(predictions_data),
                tables=[self._create_projections_table(predictions_data)],
                sources=predictions_data.get('sources', [])
            ))
            
            logger.debug(f"Created {len(sections)} PDF sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error creating PDF sections: {str(e)}")
            # Return minimal sections on error
            return [
                PDFSection(
                    title="Executive Summary",
                    content=f"Analysis report for {company_name} is being processed. Please try again later.",
                    sources=["system"]
                )
            ]
    
    def _format_executive_summary(self, company_name: str, insights: Dict) -> str:
        """Format executive summary content"""
        try:
            summary = f"""
<b>Company:</b> {company_name}<br/>
<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
<b>Analysis Confidence:</b> {insights.get('confidence', 0.5)*100:.1f}%<br/><br/>

<b>Investment Thesis:</b><br/>
{insights.get('investment_thesis', 'Investment analysis in progress.')}<br/><br/>

<b>Financial Health Score:</b> {insights.get('financial_health_score', 5.0)}/10<br/><br/>

<b>Recommendation:</b> {insights.get('recommendation', 'Hold - Analysis pending')}<br/><br/>

<b>Key Strengths:</b><br/>
"""
            
            for strength in insights.get('key_strengths', []):
                summary += f"• {strength}<br/>"
            
            summary += "<br/><b>Key Risks:</b><br/>"
            
            for risk in insights.get('key_risks', []):
                summary += f"• {risk}<br/>"
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error formatting executive summary: {str(e)}")
            return f"Executive summary for {company_name} is being processed."
    
    def _format_financial_metrics(self, financial_data: Dict) -> str:
        """Format financial metrics content"""
        try:
            if not financial_data or financial_data.get('error'):
                return "Financial metrics analysis is in progress. Data will be available shortly."
            
            metrics = financial_data.get('metrics', {})
            
            content = f"""
<b>Financial Performance Overview:</b><br/><br/>

The financial analysis reveals the following key metrics:<br/><br/>
"""
            
            if 'P/E' in metrics and metrics['P/E'] != 'N/A':
                content += f"<b>P/E Ratio:</b> {metrics['P/E']} - "
                pe_val = float(metrics['P/E']) if isinstance(metrics['P/E'], (int, float)) else 0
                if pe_val < 15:
                    content += "Potentially undervalued<br/>"
                elif pe_val > 25:
                    content += "Premium valuation<br/>"
                else:
                    content += "Fair valuation<br/>"
            
            if 'ROE' in metrics and metrics['ROE'] != 'N/A':
                content += f"<b>Return on Equity:</b> {metrics['ROE']} - "
                roe_val = float(metrics['ROE']) if isinstance(metrics['ROE'], (int, float)) else 0
                if roe_val > 0.15:
                    content += "Strong profitability<br/>"
                elif roe_val > 0.10:
                    content += "Good profitability<br/>"
                else:
                    content += "Below average profitability<br/>"
            
            content += f"<br/><b>Analysis:</b><br/>{financial_data.get('analysis', 'Detailed analysis in progress.')}"
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting financial metrics: {str(e)}")
            return "Financial metrics formatting in progress."
    
    def _create_financial_metrics_table(self, financial_data: Dict) -> Dict:
        """Create financial metrics table data"""
        try:
            if not financial_data or financial_data.get('error'):
                return {
                    "headers": ["Metric", "Value", "Status"],
                    "data": [["Analysis", "In Progress", "Pending"]],
                    "style": "financial"
                }
            
            metrics = financial_data.get('metrics', {})
            table_data = []
            
            for metric, value in metrics.items():
                if metric != 'analysis':
                    display_name = metric.replace('_', ' ').title()
                    display_value = str(value) if value != 'N/A' else 'Not Available'
                    status = "Available" if value != 'N/A' else "Pending"
                    table_data.append([display_name, display_value, status])
            
            return {
                "headers": ["Financial Metric", "Value", "Status"],
                "data": table_data if table_data else [["No Data", "Available", "Pending"]],
                "style": "financial"
            }
            
        except Exception as e:
            logger.warning(f"Error creating financial metrics table: {str(e)}")
            return {
                "headers": ["Metric", "Value", "Status"],
                "data": [["Error", "Processing", "Retry"]],
                "style": "financial"
            }
    
    def _format_business_analysis(self, business_data: Dict) -> str:
        """Format business analysis content"""
        try:
            if not business_data or business_data.get('error'):
                return "Business analysis is in progress. Strategic insights will be available shortly."
            
            content = f"""
<b>Business Model & Strategy:</b><br/><br/>

{business_data.get('analysis', 'Business strategy analysis in progress.')}<br/><br/>

<b>Sector:</b> {business_data.get('sector', 'Not specified')}<br/>
<b>Industry:</b> {business_data.get('industry', 'Not specified')}<br/><br/>

<b>Competitive Advantages:</b><br/>
"""
            
            advantages = business_data.get('competitive_advantages', [])
            if advantages:
                for advantage in advantages:
                    content += f"• {advantage}<br/>"
            else:
                content += "• Competitive analysis in progress<br/>"
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting business analysis: {str(e)}")
            return "Business analysis formatting in progress."
    
    def _format_news_analysis(self, news_data: Dict) -> str:
        """Format news analysis content"""
        try:
            if not news_data or news_data.get('error'):
                return "Market news analysis is in progress. Recent developments will be analyzed shortly."
            
            news_count = news_data.get('news_count', 0)
            overall_sentiment = news_data.get('overall_sentiment', 'Neutral')
            
            content = f"""
<b>Market News & Sentiment Analysis:</b><br/><br/>

<b>Recent News Articles Analyzed:</b> {news_count}<br/>
<b>Overall Market Sentiment:</b> {overall_sentiment}<br/><br/>

<b>Key Developments:</b><br/>
"""
            
            articles = news_data.get('articles', [])
            for i, article in enumerate(articles[:5]):  # Top 5 articles
                title = article.get('title', 'News title unavailable')
                sentiment = article.get('sentiment', 'neutral')
                content += f"{i+1}. {title} (Sentiment: {sentiment.title()})<br/>"
            
            if not articles:
                content += "Recent news articles are being processed.<br/>"
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting news analysis: {str(e)}")
            return "News analysis formatting in progress."
    
    def _create_news_summary_table(self, news_data: Dict) -> Dict:
        """Create news summary table"""
        try:
            if not news_data or news_data.get('error'):
                return {
                    "headers": ["Date", "Headline", "Sentiment"],
                    "data": [["Pending", "News analysis in progress", "Neutral"]],
                    "style": "news"
                }
            
            articles = news_data.get('articles', [])
            table_data = []
            
            for article in articles[:10]:  # Top 10 articles
                date = article.get('date', 'Unknown')
                title = article.get('title', 'Title unavailable')[:60] + "..." if len(article.get('title', '')) > 60 else article.get('title', 'Title unavailable')
                sentiment = article.get('sentiment', 'neutral').title()
                table_data.append([date, title, sentiment])
            
            return {
                "headers": ["Date", "Headline", "Sentiment"],
                "data": table_data if table_data else [["No News", "Available Currently", "Neutral"]],
                "style": "news"
            }
            
        except Exception as e:
            logger.warning(f"Error creating news summary table: {str(e)}")
            return {
                "headers": ["Date", "Headline", "Sentiment"],
                "data": [["Error", "Processing news data", "Neutral"]],
                "style": "news"
            }
    
    def _format_risk_analysis(self, risk_data: Dict, insights: Dict) -> str:
        """Format risk analysis content"""
        try:
            content = f"""
<b>Risk Assessment Overview:</b><br/><br/>

<b>Overall Risk Level:</b> {risk_data.get('overall_risk_score', 'Medium')}<br/><br/>

<b>Primary Risk Factors:</b><br/>
"""
            
            # Use insights if risk_data is not available
            key_risks = risk_data.get('top_risks', insights.get('key_risks', []))
            
            for risk in key_risks:
                content += f"• {risk}<br/>"
            
            if not key_risks:
                content += "• Risk analysis in progress<br/>"
            
            content += f"""<br/>
<b>Risk Mitigation:</b><br/>
{risk_data.get('analysis', 'Risk mitigation strategies are being analyzed.')}
"""
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting risk analysis: {str(e)}")
            return "Risk analysis formatting in progress."
    
    def _format_projections(self, predictions_data: Dict) -> str:
        """Format future projections content"""
        try:
            if not predictions_data or predictions_data.get('error'):
                return "Future projections are being calculated based on historical data and market trends."
            
            content = f"""
<b>Future Projections & Outlook:</b><br/><br/>

<b>Projection Confidence:</b> {predictions_data.get('confidence_level', 0.7)*100:.1f}%<br/><br/>

<b>Growth Assumptions:</b><br/>
{predictions_data.get('base_assumptions', {}).get('growth_rate', 'Growth rate analysis pending')}<br/><br/>

<b>Management Guidance:</b><br/>
{predictions_data.get('management_guidance', 'Management guidance analysis in progress.')}<br/><br/>

<b>Analyst Projections:</b><br/>
{predictions_data.get('analysis', 'Future outlook analysis in progress.')}
"""
            
            return content
            
        except Exception as e:
            logger.warning(f"Error formatting projections: {str(e)}")
            return "Projections formatting in progress."
    
    def _create_projections_table(self, predictions_data: Dict) -> Dict:
        """Create projections table"""
        try:
            projections = predictions_data.get('projections', [])
            
            if not projections:
                return {
                    "headers": ["Year", "Revenue (Cr)", "Growth Rate"],
                    "data": [["2025", "Projecting", "Calculating"]],
                    "style": "projections"
                }
            
            table_data = []
            for projection in projections[:5]:  # Next 5 years
                year = projection.get('year', 'Unknown')
                revenue = f"{projection.get('revenue', 0):.1f}"
                growth = projection.get('growth_rate', 'N/A')
                table_data.append([year, revenue, growth])
            
            return {
                "headers": ["Year", "Revenue (Cr)", "Growth Rate"],
                "data": table_data,
                "style": "projections"
            }
            
        except Exception as e:
            logger.warning(f"Error creating projections table: {str(e)}")
            return {
                "headers": ["Year", "Revenue (Cr)", "Growth Rate"],
                "data": [["Error", "Processing", "Data"]],
                "style": "projections"
            }

    async def _create_pdf_document(self, output_path: str, company_name: str, sections: List[PDFSection], data_quality: Dict) -> Dict[str, Any]:
        """Create the actual PDF document"""
        logger.debug(f"Creating PDF document at {output_path}")
        
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story
            story = []
            
            # Title page
            story.append(Paragraph(f"Financial Analysis Report", self.styles['CustomTitle']))
            story.append(Paragraph(f"{company_name}", self.styles['CustomTitle']))
            story.append(Spacer(1, 24))
            
            # Data quality indicator
            quality_score = data_quality.get('overall_quality', 0.0) * 100
            quality_color = 'green' if quality_score > 80 else 'orange' if quality_score > 60 else 'red'
            story.append(Paragraph(f'<font color="{quality_color}">Data Quality: {quality_score:.1f}%</font>', self.styles['CustomBody']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['CustomBody']))
            story.append(Spacer(1, 36))
            
            # Add sections
            for section in sections:
                # Section title
                story.append(Paragraph(section.title, self.styles['CustomHeading']))
                story.append(Spacer(1, 12))
                
                # Section content
                story.append(Paragraph(section.content, self.styles['CustomBody']))
                story.append(Spacer(1, 12))
                
                # Add tables if present
                if section.tables:
                    for table_data in section.tables:
                        table = self._create_table(table_data)
                        if table:
                            story.append(table)
                            story.append(Spacer(1, 12))
                
                # Add sources
                if section.sources:
                    sources_text = f"<i>Sources: {', '.join(section.sources[:5])}</i>"  # Limit to 5 sources
                    story.append(Paragraph(sources_text, self.styles['CustomBullet']))
                
                story.append(Spacer(1, 24))
            
            # Footer
            story.append(PageBreak())
            story.append(Paragraph("Disclaimer", self.styles['CustomHeading']))
            story.append(Paragraph(
                "This report is generated by AlphaSage AI system for informational purposes only. "
                "It should not be considered as investment advice. Please consult with qualified "
                "financial advisors before making investment decisions.", 
                self.styles['CustomBody']
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.debug("PDF document created successfully")
            return {
                "success": True,
                "file_size": os.path.getsize(output_path),
                "sections_count": len(sections)
            }
            
        except Exception as e:
            logger.error(f"Error creating PDF document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_table(self, table_data: Dict) -> Optional[Table]:
        """Create a formatted table for the PDF"""
        try:
            headers = table_data.get('headers', [])
            data = table_data.get('data', [])
            style_name = table_data.get('style', 'default')
            
            if not headers or not data:
                return None
            
            # Prepare table data
            table_list = [headers] + data
            
            # Create table
            table = Table(table_list, repeatRows=1)
            
            # Apply styles based on table type
            if style_name == 'financial':
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f5f8f')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
            elif style_name == 'news':
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
            else:  # default and projections
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5cb85c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
            
            table.setStyle(table_style)
            return table
            
        except Exception as e:
            logger.warning(f"Error creating table: {str(e)}")
            return None

# Example usage and testing
async def test_pdf_agent():
    """Test PDF agent functionality"""
    try:
        # Create PDF agent
        pdf_agent = AlphaSagePDFAgent()
        
        # Test company
        test_company = "Ganesha Ecosphere Limited"
        
        print(f"Generating PDF report for {test_company}...")
        
        # Generate report
        result = await pdf_agent.generate_comprehensive_report(test_company)
        
        if result['success']:
            print(f"✓ PDF report generated successfully!")
            print(f"  File: {result['output_path']}")
            print(f"  Size: {result['file_size']} bytes")
            print(f"  Sections: {result['sections_count']}")
            print(f"  Data Quality: {result['data_quality']['overall_quality']*100:.1f}%")
            print(f"  Time: {result['execution_time']:.2f}s")
        else:
            print(f"✗ PDF generation failed: {result['error']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pdf_agent())