import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any

from dotenv import load_dotenv
from pymongo import MongoClient
import yfinance as yf

from vector import AlphaSageVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_test_data.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv('alphasage.env')

# Test configuration
TEST_COMPANY = "Ganesha Ecosphere Limited"
TEST_TICKER = "GANECOS.NS"
MIN_CHUNKS = 20

async def load_test_data():
    """Load test data for Ganesha Ecosphere Limited"""
    logging.info("Starting test data load...")
    mongo_client = None
    vector_db = None
    
    try:
        # Initialize MongoDB and ChromaDB
        mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        logging.info("MongoDB connection established")
        
        vector_db = AlphaSageVectorDB()
        logging.info("Vector DB initialized")
        
        # Get company data from Yahoo Finance
        logging.info(f"Fetching data for {TEST_TICKER} from Yahoo Finance")
        stock = yf.Ticker(TEST_TICKER)
        
        # Create chunks for different categories
        chunks = []
        
        # 1. Company Info
        logging.info("Fetching company info")
        info = stock.info
        if not info:
            raise ValueError(f"No company info found for {TEST_TICKER}")
            
        chunks.append({
            "company_name": TEST_COMPANY,
            "document_date": datetime.now().strftime("%Y-%m-%d"),
            "category": "Company Info",
            "content": {
                "type": "text",
                "text": f"Ganesha Ecosphere Limited is a {info.get('sector', '')} company in the {info.get('industry', '')} industry. "
                       f"The company has a market cap of {info.get('marketCap', '')} and {info.get('fullTimeEmployees', '')} employees. "
                       f"Its headquarters are located in {info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}.",
                "keywords": ["company", "info", "overview", "description"]
            },
            "source": "Yahoo Finance"
        })
        
        # 2. Financial Ratios
        logging.info("Creating financial ratios chunk")
        ratios = {
            "P/E Ratio": info.get('trailingPE', ''),
            "PEG Ratio": info.get('pegRatio', ''),
            "Price to Book": info.get('priceToBook', ''),
            "Price to Sales": info.get('priceToSalesTrailing12Months', ''),
            "Dividend Yield": info.get('dividendYield', ''),
            "ROE": info.get('returnOnEquity', ''),
            "ROA": info.get('returnOnAssets', ''),
            "Profit Margin": info.get('profitMargins', '')
        }
        
        chunks.append({
            "company_name": TEST_COMPANY,
            "document_date": datetime.now().strftime("%Y-%m-%d"),
            "category": "Financial Ratios",
            "content": {
                "type": "table",
                "text": "Key Financial Ratios",
                "table": [
                    {"row": i, "column": 0, "value": ratio}
                    for i, ratio in enumerate(ratios.keys())
                ] + [
                    {"row": i, "column": 1, "value": str(value)}
                    for i, value in enumerate(ratios.values())
                ],
                "keywords": ["ratios", "financial", "metrics", "valuation"]
            },
            "source": "Yahoo Finance"
        })
        
        # 3. Historical Data
        logging.info("Fetching historical data")
        hist = stock.history(period="1y")
        if hist.empty:
            raise ValueError(f"No historical data found for {TEST_TICKER}")
            
        chunks.append({
            "company_name": TEST_COMPANY,
            "document_date": datetime.now().strftime("%Y-%m-%d"),
            "category": "Historical Data",
            "content": {
                "type": "table",
                "text": "Historical Stock Prices",
                "table": [
                    {"row": i, "column": 0, "value": str(date)}
                    for i, date in enumerate(hist.index)
                ] + [
                    {"row": i, "column": 1, "value": str(hist['Close'][i])}
                    for i in range(len(hist))
                ],
                "keywords": ["historical", "prices", "stock", "performance"]
            },
            "source": "Yahoo Finance"
        })
        
        # 4. News Articles
        logging.info("Fetching news articles")
        news = stock.news
        if not news:
            logging.warning(f"No news articles found for {TEST_TICKER}")
        else:
            for article in news[:5]:  # Get last 5 news articles
                chunks.append({
                    "company_name": TEST_COMPANY,
                    "document_date": datetime.fromtimestamp(article['providerPublishTime']).strftime("%Y-%m-%d"),
                    "category": "News",
                    "content": {
                        "type": "text",
                        "text": f"{article['title']}\n\n{article.get('summary', '')}",
                        "keywords": ["news", "article", "update", "announcement"]
                    },
                    "source": article.get('publisher', 'Yahoo Finance')
                })
        
        # 5. Financial Statements
        logging.info("Fetching financial statements")
        income_stmt = stock.financials
        if income_stmt.empty:
            logging.warning(f"No financial statements found for {TEST_TICKER}")
        else:
            chunks.append({
                "company_name": TEST_COMPANY,
                "document_date": datetime.now().strftime("%Y-%m-%d"),
                "category": "Financial Statements",
                "content": {
                    "type": "table",
                    "text": "Income Statement",
                    "table": [
                        {"row": i, "column": 0, "value": str(date)}
                        for i, date in enumerate(income_stmt.columns)
                    ] + [
                        {"row": i, "column": 1, "value": str(income_stmt.iloc[0][i])}
                        for i in range(len(income_stmt.columns))
                    ],
                    "keywords": ["financial", "income", "revenue", "statement"]
                },
                "source": "Yahoo Finance"
            })
        
        # Load chunks into MongoDB
        logging.info(f"Loading {len(chunks)} chunks into MongoDB")
        db = mongo_client.alphasage
        collection = db.alphasage_chunks
        
        # Clear existing chunks for test company
        delete_result = collection.delete_many({"company_name": TEST_COMPANY})
        logging.info(f"Deleted {delete_result.deleted_count} existing chunks")
        
        # Insert new chunks
        result = collection.insert_many(chunks)
        logging.info(f"Inserted {len(result.inserted_ids)} chunks into MongoDB")
        
        # Embed chunks in ChromaDB
        logging.info("Starting batch embedding")
        embedding_stats = vector_db.embed_chunks_batch(batch_size=100, skip_existing=False)
        logging.info(f"Embedding stats: {embedding_stats}")
        
        # Verify chunks
        chunks = vector_db.retrieve_chunks(company_name=TEST_COMPANY)
        logging.info(f"Retrieved {len(chunks)} chunks from ChromaDB")
        
        if len(chunks) < MIN_CHUNKS:
            logging.warning(f"Only {len(chunks)} chunks found, expected at least {MIN_CHUNKS}")
        else:
            logging.info("Test data load completed successfully")
        
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        if mongo_client:
            mongo_client.close()
            logging.info("MongoDB connection closed")
        if vector_db:
            vector_db.close()
            logging.info("Vector DB connection closed")

if __name__ == "__main__":
    asyncio.run(load_test_data()) 