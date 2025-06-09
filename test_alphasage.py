import asyncio
import logging
import os
import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
import redis
from vector import AlphaSageVectorDB
from tools import (
    ReasoningTool, YFinanceNumberTool, YFinanceNewsTool,
    ArithmeticCalculationTool, VectorSearchRAGTool
)
from microagent import AlphaSageMicroAgents
from macroagent import AlphaSageMacroAgents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_alphasage.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv('alphasage.env')

# Test configuration
TEST_COMPANY = "Ganesha Ecosphere Limited"
TEST_TICKER = "GANECOS.NS"
TEST_YEARS = [2023, 2024, 2025]
MIN_CHUNKS = 20

class TestAlphaSage:
    """Comprehensive test suite for AlphaSage components"""

    def __init__(self):
        """Initialize test environment"""
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379))
        )
        self.vector_db = AlphaSageVectorDB()
        self.micro_agents = AlphaSageMicroAgents()
        self.macro_agents = AlphaSageMacroAgents()
        
        # Initialize test data
        self.test_data = {
            'company_name': TEST_COMPANY,
            'ticker': TEST_TICKER,
            'years': TEST_YEARS
        }

    async def setup(self):
        """Setup test environment"""
        try:
            # Verify MongoDB connection
            self.mongo_client.admin.command('ping')
            logging.info("MongoDB connection established")
            
            # Verify Redis connection
            self.redis_client.ping()
            logging.info("Redis connection established")
            
            # Verify ChromaDB connection
            self.vector_db._initialize_chromadb_collection()
            logging.info("ChromaDB connection established")
            
            # Verify minimum chunks exist
            chunks = self.vector_db.retrieve_chunks(company_name=TEST_COMPANY)
            if len(chunks) < MIN_CHUNKS:
                raise ValueError(f"Expected at least {MIN_CHUNKS} chunks for {TEST_COMPANY}")
            
            return True
        except Exception as e:
            logging.error(f"Setup failed: {str(e)}")
            return False

    async def cleanup(self):
        """Cleanup test environment"""
        try:
            self.mongo_client.close()
            self.redis_client.close()
            self.vector_db.close()
            logging.info("Test environment cleaned up")
        except Exception as e:
            logging.error(f"Cleanup failed: {str(e)}")

    async def test_vector_db(self):
        """Test vector database functionality"""
        try:
            # Test basic retrieval
            chunks = self.vector_db.retrieve_chunks(
                company_name=TEST_COMPANY,
                n_results=5
            )
            assert len(chunks) > 0, "No chunks retrieved"
            
            # Test metadata filtering
            filtered_chunks = self.vector_db.retrieve_chunks(
                company_name=TEST_COMPANY,
                category="Valuation Ratios",
                n_results=5
            )
            assert len(filtered_chunks) > 0, "No filtered chunks retrieved"
            
            # Test semantic search
            search_results = self.vector_db.search_similar_chunks(
                "revenue growth and profitability",
                n_results=5
            )
            assert len(search_results) > 0, "No semantic search results"
            
            logging.info("Vector database tests passed")
            return True
        except Exception as e:
            logging.error(f"Vector database test failed: {str(e)}")
            return False

    async def test_tools(self):
        """Test utility tools"""
        try:
            # Test YFinanceNumberTool
            pe_ratio = YFinanceNumberTool.get_financial_ratio_by_date(
                TEST_TICKER,
                "PE Ratio",
                datetime.now().strftime("%Y-%m-%d")
            )
            assert pe_ratio['value'] > 0, "Invalid P/E ratio"
            
            # Test YFinanceNewsTool
            news = await YFinanceNewsTool.fetch_company_news(TEST_TICKER)
            assert len(news) > 0, "No news retrieved"
            
            # Test ReasoningTool
            reasoning = await ReasoningTool.reason_on_data(
                pe_ratio,
                "Analyze the P/E ratio trend"
            )
            assert 'reasoning' in reasoning, "No reasoning generated"
            
            # Test VectorSearchRAGTool
            rag_results = await VectorSearchRAGTool.search_knowledge_base(
                "revenue growth",
                TEST_COMPANY
            )
            assert len(rag_results) > 0, "No RAG results"
            
            logging.info("Utility tools tests passed")
            return True
        except Exception as e:
            logging.error(f"Tools test failed: {str(e)}")
            return False

    async def test_micro_agents(self):
        """Test micro agents"""
        try:
            # Test valuation ratios
            valuation = await self.micro_agents.calculate_valuation_ratios(
                TEST_TICKER,
                TEST_COMPANY
            )
            assert valuation.success, "Valuation ratios calculation failed"
            
            # Test profitability ratios
            profitability = await self.micro_agents.calculate_profitability_ratios(
                TEST_TICKER,
                TEST_COMPANY
            )
            assert profitability.success, "Profitability ratios calculation failed"
            
            # Test news analysis
            news = await self.micro_agents.analyze_news(TEST_TICKER)
            assert news.success, "News analysis failed"
            
            logging.info("Micro agents tests passed")
            return True
        except Exception as e:
            logging.error(f"Micro agents test failed: {str(e)}")
            return False

    async def test_macro_agents(self):
        """Test macro agents"""
        try:
            # Test business analysis
            business = await self.macro_agents.analyze_business(TEST_COMPANY)
            assert business.success, "Business analysis failed"
            
            # Test future predictions
            future = await self.macro_agents.predict_future(TEST_COMPANY)
            assert future.success, "Future predictions failed"
            
            # Test risk analysis
            risks = await self.macro_agents.analyze_risks(TEST_COMPANY)
            assert risks.success, "Risk analysis failed"
            
            logging.info("Macro agents tests passed")
            return True
        except Exception as e:
            logging.error(f"Macro agents test failed: {str(e)}")
            return False

    async def test_integration(self):
        """Test end-to-end integration"""
        try:
            # Test complete analysis flow
            business = await self.macro_agents.analyze_business(TEST_COMPANY)
            assert business.success, "Business analysis failed"
            
            # Test batch processing
            inputs = [
                {"company_name": TEST_COMPANY, "ticker": TEST_TICKER}
            ]
            results = await self.macro_agents.batch_orchestrate(
                self.macro_agents.analyze_business,
                inputs
            )
            assert len(results) > 0, "Batch processing failed"
            
            logging.info("Integration tests passed")
            return True
        except Exception as e:
            logging.error(f"Integration test failed: {str(e)}")
            return False

async def main():
    """Main test runner"""
    logging.info("Starting AlphaSage test suite...")
    
    test_suite = TestAlphaSage()
    
    # Run setup
    if not await test_suite.setup():
        logging.error("Test setup failed")
        return
    
    try:
        # Run all tests
        tests = [
            test_suite.test_vector_db(),
            test_suite.test_tools(),
            test_suite.test_micro_agents(),
            test_suite.test_macro_agents(),
            test_suite.test_integration()
        ]
        
        results = await asyncio.gather(*tests)
        
        # Log results
        for test_name, result in zip(
            ["Vector DB", "Tools", "Micro Agents", "Macro Agents", "Integration"],
            results
        ):
            status = "PASSED" if result else "FAILED"
            logging.info(f"{test_name} tests: {status}")
        
    finally:
        # Cleanup
        await test_suite.cleanup()
    
    logging.info("AlphaSage test suite completed")

if __name__ == "__main__":
    asyncio.run(main()) 