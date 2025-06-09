import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
import pymongo
from pymongo import MongoClient
import chromadb
import yfinance as yf
import redis
from dotenv import load_dotenv
import autogen

# Import from macroagent
from macroagent import AlphaSageMacroAgents, MacroAgentResult

# Load environment variables
load_dotenv('alphasage.env')  # Updated to use the correct .env file

# Constants
TODAY = datetime.now()
TODAY_STR = TODAY.strftime("%Y-%m-%d")
COMPANY_TICKER = "GANECOS.NS"
COMPANY_NAME = "Ganesha Ecosphere Limited"
OUTPUT_DIR = "output"
LOG_FILE = "alphasage.log"
TEST_LOG_FILE = "test_alphasage.log"

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "alphasage_chunks"
COLLECTION_NAME = "chunks"

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_EXPIRY = 3600  # 1 hour

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PDFOutputAgent')

class PDFOutputAgent:
    """AutoGen agent for generating structured PDF reports from AlphaSage insights."""
    
    def __init__(self):
        """Initialize the PDF Output Agent with required connections and tools."""
        # Initialize PDF styles
        self.styles = self._create_styles()
        
        # Initialize connections
        self.mongo_client = MongoClient(MONGODB_URI)
        self.db = self.mongo_client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        logger.info("MongoDB connection established")
        
        self.redis_client = redis.from_url(REDIS_URL)
        logger.info("Redis connection established")
        
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("alphasage")
        logger.info("ChromaDB connection established")
        
        self.ticker = yf.Ticker(COMPANY_TICKER)
        
        # Initialize macro agents
        self.macro_agents = AlphaSageMacroAgents()
        
        # Create AutoGen agent
        self.agent = self._create_autogen_agent()
    
    def _create_styles(self) -> Dict:
        """Create custom PDF styles."""
        styles = getSampleStyleSheet()
        
        # Add custom styles
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                name='CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            ),
            'SectionHeader': ParagraphStyle(
                name='SectionHeader',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            ),
            'SubHeader': ParagraphStyle(
                name='SubHeader',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10
            ),
            'CustomBodyText': ParagraphStyle(
                name='CustomBodyText',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            ),
            'Footnote': ParagraphStyle(
                name='Footnote',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.gray
            )
        }
        
        # Add styles to stylesheet
        for name, style in custom_styles.items():
            styles.add(style)
        
        return styles
    
    def _create_autogen_agent(self) -> autogen.ConversableAgent:
        """Create and configure the AutoGen agent with Groq."""
        config_list = [{
            "model": "mixtral-8x7b-32768",  # Using Mixtral model via Groq
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": "https://api.groq.com/v1",
            "api_type": "groq"
        }]
        
        return autogen.ConversableAgent(
            name="PDFOutputAgent",
            llm_config={
                "config_list": config_list,
                "temperature": 0.2,
                "max_tokens": 2000,
                "cache_seed": 42
            },
            system_message=f"""You are an expert financial analyst assistant specialized in creating structured PDF reports.
Today is {TODAY_STR}. Your task is to generate comprehensive PDF reports for {COMPANY_NAME}.
Focus on accuracy, traceability, and professional formatting."""
        )
    
    async def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Retrieve cached data from Redis."""
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Redis cache error: {str(e)}")
            return None
    
    async def _cache_data(self, key: str, data: Dict, expiry: int = REDIS_EXPIRY):
        """Cache data in Redis."""
        try:
            self.redis_client.setex(key, expiry, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis cache error: {str(e)}")
    
    async def _get_mongodb_chunks(self, category: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant chunks from MongoDB."""
        try:
            query = {"company_name": COMPANY_NAME}
            if category:
                query["category"] = category
            return list(self.collection.find(query))
        except Exception as e:
            logger.error(f"MongoDB query error: {str(e)}")
            return []
    
    async def _get_chromadb_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant chunks from ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB query error: {str(e)}")
            return []
    
    def _create_cover_page(self, canvas, doc):
        """Create the cover page with company name, date, and logo placeholder."""
        canvas.saveState()
        
        # Company Name
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(doc.width/2, doc.height - 2*inch, COMPANY_NAME)
        
        # Date
        canvas.setFont("Helvetica", 12)
        canvas.drawCentredString(doc.width/2, doc.height - 3*inch, f"Report Generated: {TODAY_STR}")
        
        # Logo Placeholder
        canvas.rect(doc.width/2 - 1*inch, doc.height - 5*inch, 2*inch, 2*inch)
        canvas.drawCentredString(doc.width/2, doc.height - 4*inch, "Company Logo")
        
        # Footer
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(doc.width/2, 0.5*inch, "AlphaSage Financial Intelligence Platform")
        
        canvas.restoreState()
    
    def _create_overview_section(self, doc, data: Dict) -> List:
        """Create the Overview section with company description and key metrics."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Overview", self.styles['SectionHeader']))
        
        # Company Description
        elements.append(Paragraph(data.get("description", ""), self.styles['CustomBodyText']))
        
        # Key Metrics as text
        elements.append(Paragraph("Key Metrics:", self.styles['SubHeader']))
        metrics_text = f"""
        Market Cap: {data.get("market_cap", "N/A")}
        Sector: {data.get("sector", "N/A")}
        Industry: {data.get("industry", "N/A")}
        """
        elements.append(Paragraph(metrics_text, self.styles['CustomBodyText']))
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_company_structure_section(self, doc, data: Dict) -> List:
        """Create the Company Structure section with management and subsidiaries."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Company Structure", self.styles['SectionHeader']))
        
        # Management Team
        elements.append(Paragraph("Management Team", self.styles['SubHeader']))
        management_text = "\n".join([f"• {member['name']} - {member['position']}" for member in data.get("management", [])])
        elements.append(Paragraph(management_text, self.styles['CustomBodyText']))
        
        # Subsidiaries
        elements.append(Paragraph("Subsidiaries", self.styles['SubHeader']))
        subsidiaries_text = "\n".join([f"• {subsidiary['name']} ({subsidiary['ownership']}%)" for subsidiary in data.get("subsidiaries", [])])
        elements.append(Paragraph(subsidiaries_text, self.styles['CustomBodyText']))
        
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_news_section(self, doc, data: Dict) -> List:
        """Create the News section with recent news and sentiment analysis."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Recent News", self.styles['SectionHeader']))
        
        # News as text
        for news in data.get("news", []):
            news_text = f"""
            Date: {news.get("date", "N/A")}
            Title: {news.get("title", "N/A")}
            Sentiment: {news.get("sentiment", "N/A")}
            Source: {news.get("source", "N/A")}
            """
            elements.append(Paragraph(news_text, self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_financials_section(self, doc, data: Dict) -> List:
        """Create the Current Financials section with ratios and share price."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Current Financials", self.styles['SectionHeader']))
        
        # Ratios as text
        elements.append(Paragraph("Financial Ratios:", self.styles['SubHeader']))
        for ratio in data.get("ratios", []):
            ratio_text = f"""
            {ratio.get("name", "N/A")}:
            Value: {ratio.get("value", "N/A")}
            Industry Average: {ratio.get("industry_avg", "N/A")}
            Source: {ratio.get("source", "N/A")}
            """
            elements.append(Paragraph(ratio_text, self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.1*inch))
        
        # Share Price as text
        if data.get("share_price_data"):
            elements.append(Paragraph("Share Price History:", self.styles['SubHeader']))
            share_price_text = f"""
            Latest Price: {data["share_price_data"][-1] if data["share_price_data"] else "N/A"}
            Date: {data["share_price_dates"][-1] if data["share_price_dates"] else "N/A"}
            """
            elements.append(Paragraph(share_price_text, self.styles['CustomBodyText']))
        
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_shareholdings_section(self, doc, data: Dict) -> List:
        """Create the Shareholdings section with ownership breakdown."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Shareholdings", self.styles['SectionHeader']))
        
        # Shareholdings as text
        for holding in data.get("shareholdings", []):
            holding_text = f"""
            Category: {holding.get("category", "N/A")}
            Percentage: {holding.get("percentage", "N/A")}%
            Source: {holding.get("source", "N/A")}
            """
            elements.append(Paragraph(holding_text, self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_past_promises_section(self, doc, data: Dict) -> List:
        """Create the Past Promises section with guidance vs. actuals."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Past Promises", self.styles['SectionHeader']))
        
        # Promises as text
        for promise in data.get("promises", []):
            promise_text = f"""
            Metric: {promise.get("metric", "N/A")}
            Guidance: {promise.get("guidance", "N/A")}
            Actual: {promise.get("actual", "N/A")}
            Variance: {promise.get("variance", "N/A")}
            Source: {promise.get("source", "N/A")}
            """
            elements.append(Paragraph(promise_text, self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_future_insights_section(self, doc, data: Dict) -> List:
        """Create the Future Insights section with projections and risks."""
        elements = []
        
        # Section Header
        elements.append(Paragraph("Future Insights", self.styles['SectionHeader']))
        
        # Projections as text
        elements.append(Paragraph("Projections:", self.styles['SubHeader']))
        for projection in data.get("projections", []):
            projection_text = f"""
            Metric: {projection.get("metric", "N/A")}
            FY2026: {projection.get("fy2026", "N/A")}
            FY2027: {projection.get("fy2027", "N/A")}
            FY2028: {projection.get("fy2028", "N/A")}
            Source: {projection.get("source", "N/A")}
            """
            elements.append(Paragraph(projection_text, self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.1*inch))
        
        # Risks as text
        elements.append(Paragraph("Key Risks:", self.styles['SubHeader']))
        risks_text = "\n".join([f"• {risk.get('description', 'N/A')}" for risk in data.get("risks", [])])
        elements.append(Paragraph(risks_text, self.styles['CustomBodyText']))
        
        return elements
    
    async def generate_pdf_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive PDF report for the company."""
        if not output_path:
            output_path = os.path.join(OUTPUT_DIR, f"{COMPANY_NAME.replace(' ', '_')}_report.pdf")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Collect data from macro agents
            analysis_results = await self.macro_agents.orchestrate_comprehensive_analysis(COMPANY_NAME)
            
            # Extract data for PDF sections
            data = {
                "description": analysis_results.business.data.get("business_model", "") if hasattr(analysis_results, 'business') else "",
                "market_cap": self.ticker.info.get("marketCap"),
                "sector": self.ticker.info.get("sector"),
                "industry": self.ticker.info.get("industry"),
                "management": analysis_results.deepdive.data.get("management", []) if hasattr(analysis_results, 'deepdive') else [],
                "subsidiaries": analysis_results.deepdive.data.get("subsidiaries", []) if hasattr(analysis_results, 'deepdive') else [],
                "news": analysis_results.current_affairs.data.get("events", []) if hasattr(analysis_results, 'current_affairs') else [],
                "ratios": analysis_results.debt_wc.data.get("ratios", []) if hasattr(analysis_results, 'debt_wc') else [],
                "share_price_data": self.ticker.history(period="1y")["Close"].tolist(),
                "share_price_dates": self.ticker.history(period="1y").index.strftime("%Y-%m-%d").tolist(),
                "shareholdings": self.ticker.info.get("majorHolders", []),
                "promises": analysis_results.concall.data.get("guidance", []) if hasattr(analysis_results, 'concall') else [],
                "projections": analysis_results.predictions.data.get("projections", []) if hasattr(analysis_results, 'predictions') else [],
                "risks": analysis_results.risks.data.get("risks", []) if hasattr(analysis_results, 'risks') else []
            }
            
            # Build PDF content
            elements = []
            
            # Add sections
            elements.extend(self._create_overview_section(doc, data))
            elements.extend(self._create_company_structure_section(doc, data))
            elements.extend(self._create_news_section(doc, data))
            elements.extend(self._create_financials_section(doc, data))
            elements.extend(self._create_shareholdings_section(doc, data))
            elements.extend(self._create_past_promises_section(doc, data))
            elements.extend(self._create_future_insights_section(doc, data))
            
            # Build PDF
            doc.build(elements, onFirstPage=self._create_cover_page)
            
            logger.info(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise

async def main():
    """Main function to run the PDF Output Agent."""
    agent = PDFOutputAgent()
    output_path = await agent.generate_pdf_report()
    print(f"PDF report generated: {output_path}")

if __name__ == "__main__":
    asyncio.run(main()) 