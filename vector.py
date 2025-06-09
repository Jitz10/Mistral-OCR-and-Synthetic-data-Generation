import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import traceback

# Check for required dependencies
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print("ERROR: Missing chromadb library. Please install it with:")
    print("pip install chromadb")
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

@dataclass
class ChunkData:
    """Data structure for processed chunks"""
    id: str
    text: str
    metadata: Dict[str, Any]
    mongodb_id: str

class AlphaSageVectorDB:
    """ChromaDB vector database for AlphaSage financial intelligence platform"""
    
    def __init__(self, chromadb_path: str = "./chromadb_data", collection_name: str = "alphasage_chunks"):
        """Initialize the vector database with ChromaDB and MongoDB connections"""
        
        # Initialize Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.mongo_client['alphasage']
        self.collection = self.db['alphasage_chunks']
        
        # Create MongoDB indexes for efficient querying
        self._create_mongodb_indexes()
        
        # Initialize ChromaDB
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(
            path=chromadb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get ChromaDB collection
        self.chroma_collection = self._initialize_chromadb_collection()
        
        # Rate limiting for Gemini API
        self.last_api_call = 0
        self.api_call_delay = 1.0  # 1 second between calls
        
        logger.info(f"AlphaSageVectorDB initialized with ChromaDB at {chromadb_path}")

    def _create_mongodb_indexes(self):
        """Create MongoDB indexes for efficient querying"""
        try:
            # Create compound indexes for common query patterns
            self.collection.create_index([("company_name", 1), ("document_date", 1)])
            self.collection.create_index([("company_name", 1), ("category", 1)])
            self.collection.create_index([("category", 1), ("document_date", 1)])
            self.collection.create_index("company_name")
            self.collection.create_index("category")
            self.collection.create_index("document_date")
            self.collection.create_index("source")
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    def _initialize_chromadb_collection(self):
        """Initialize or get existing ChromaDB collection"""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            # Create new collection with custom embedding function
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # We'll handle embeddings manually
                metadata={"description": "AlphaSage financial intelligence chunks"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
        
        return collection

    def _rate_limit_api_call(self):
        """Implement rate limiting for Gemini API calls"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.api_call_delay:
            sleep_time = self.api_call_delay - elapsed
            time.sleep(sleep_time)
        self.last_api_call = time.time()

    def _generate_embedding(self, text: str, retries: int = 3) -> Optional[List[float]]:
        """Generate embedding using Google Gemini embedding-001 model"""
        if not text.strip():
            return None
        
        for attempt in range(retries):
            try:
                self._rate_limit_api_call()
                
                # Use Gemini embedding model
                result = genai.embed_content(
                    model="embedding-001",
                    content=text[:8192],  # Limit text length for API
                    task_type="retrieval_document"
                )
                
                if result and 'embedding' in result:
                    return result['embedding']
                else:
                    logger.warning(f"No embedding returned for text: {text[:100]}...")
                    return None
                    
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.api_call_delay
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {retries} attempts: {e}")
                    return None
        
        return None

    def _process_chunk_content(self, chunk: Dict[str, Any]) -> str:
        """Process chunk content to extract text for embedding"""
        content = chunk.get('content', {})
        content_type = content.get('type', 'text')
        
        if content_type == 'text':
            text = content.get('text', '').strip()
            if text:
                return text
        
        # Handle table content
        table_data = content.get('table', [])
        if table_data:
            # Convert table to text summary
            table_text = self._table_to_text(table_data)
            if table_text:
                return table_text
        
        # Fallback to keywords if available
        keywords = content.get('keywords', [])
        if keywords:
            return ' '.join(keywords)
        
        return ""

    def _table_to_text(self, table_data: List[Dict[str, Any]]) -> str:
        """Convert table data to text summary for embedding"""
        if not table_data:
            return ""
        
        # Extract cell values and create a concise summary
        values = []
        for cell in table_data[:20]:  # Limit to first 20 cells to keep concise
            if isinstance(cell, dict) and 'value' in cell:
                value = str(cell['value']).strip()
                if value and value not in values:
                    values.append(value)
        
        return ' '.join(values)

    def _create_chunk_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata dictionary for ChromaDB storage"""
        return {
            'company_name': chunk.get('company_name', ''),
            'document_date': chunk.get('document_date', ''),
            'category': chunk.get('category', ''),
            'source': chunk.get('source', ''),
            'content_type': chunk.get('content', {}).get('type', 'text'),
            'chunk_id': chunk.get('chunk_id', ''),
            'mongodb_id': str(chunk.get('_id', ''))
        }

    def _chunk_exists_in_chromadb(self, mongodb_id: str) -> bool:
        """Check if chunk already exists in ChromaDB"""
        try:
            results = self.chroma_collection.get(
                where={"mongodb_id": mongodb_id},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.warning(f"Error checking chunk existence: {e}")
            return False

    def embed_chunks_batch(self, batch_size: int = 100, skip_existing: bool = True) -> Dict[str, int]:
        """Embed chunks from MongoDB in batches and store in ChromaDB"""
        stats = {
            'processed': 0,
            'embedded': 0,
            'skipped': 0,
            'errors': 0
        }
        
        try:
            # Get total count for progress tracking
            total_chunks = self.collection.count_documents({})
            logger.info(f"Starting batch embedding of {total_chunks} chunks")
            
            # Process chunks in batches
            skip = 0
            while skip < total_chunks:
                batch = list(self.collection.find({}).skip(skip).limit(batch_size))
                if not batch:
                    break
                
                logger.info(f"Processing batch {skip//batch_size + 1}, chunks {skip+1}-{skip+len(batch)}")
                
                # Process each chunk in the batch
                embeddings = []
                metadatas = []
                ids = []
                documents = []
                
                for chunk in batch:
                    try:
                        stats['processed'] += 1
                        mongodb_id = str(chunk['_id'])
                        
                        # Skip if already exists and skip_existing is True
                        if skip_existing and self._chunk_exists_in_chromadb(mongodb_id):
                            stats['skipped'] += 1
                            continue
                        
                        # Extract text content
                        text_content = self._process_chunk_content(chunk)
                        if not text_content:
                            logger.warning(f"No text content found for chunk {mongodb_id}")
                            continue
                        
                        # Generate embedding
                        embedding = self._generate_embedding(text_content)
                        if embedding is None:
                            stats['errors'] += 1
                            continue
                        
                        # Prepare data for batch insert
                        chunk_id = f"chunk_{mongodb_id}"
                        metadata = self._create_chunk_metadata(chunk)
                        
                        embeddings.append(embedding)
                        metadatas.append(metadata)
                        ids.append(chunk_id)
                        documents.append(text_content[:1000])  # Store truncated text
                        
                        stats['embedded'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.get('_id')}: {e}")
                        stats['errors'] += 1
                
                # Batch insert into ChromaDB
                if embeddings:
                    try:
                        self.chroma_collection.add(
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=documents,
                            ids=ids
                        )
                        logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")
                    except Exception as e:
                        logger.error(f"Error adding batch to ChromaDB: {e}")
                        stats['errors'] += len(embeddings)
                
                skip += batch_size
                
                # Progress update
                if skip % (batch_size * 5) == 0:
                    logger.info(f"Progress: {skip}/{total_chunks} chunks processed")
            
            logger.info(f"Batch embedding completed. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            logger.error(traceback.format_exc())
            return stats

    def retrieve_chunks(self, 
                       company_name: Optional[str] = None,
                       category: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       query_text: Optional[str] = None,
                       n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with metadata filtering and optional semantic search
        
        Args:
            company_name: Filter by company name
            category: Filter by category (e.g., "Future Insights")
            start_date: Filter by start date (YYYY-MM-DD format)
            end_date: Filter by end date (YYYY-MM-DD format)
            query_text: Optional semantic search query
            n_results: Maximum number of results to return
        
        Returns:
            List of matching chunks from MongoDB
        """
        try:
            # Build metadata filter with proper ChromaDB syntax
            where_filter = {}
            
            # ChromaDB requires proper boolean logic for multiple conditions
            if company_name:
                where_filter["company_name"] = {"$eq": company_name}
            if category:
                where_filter["category"] = {"$eq": category}
            
            if query_text:
                # Semantic search with query embedding
                query_embedding = self._generate_embedding(query_text)
                if query_embedding is None:
                    logger.error("Failed to generate embedding for query text")
                    return []
                
                # Query ChromaDB with semantic search
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    where=where_filter,
                    n_results=min(n_results * 2, 50)  # Get more results for date filtering
                )
            else:
                # Metadata-only search
                results = self.chroma_collection.get(
                    where=where_filter,
                    limit=min(n_results * 2, 50)
                )
            
            # Handle different result structures from query vs get
            if 'ids' not in results or not results['ids']:
                logger.info("No results found in ChromaDB")
                return []
            
            # Extract MongoDB IDs from ChromaDB results
            mongodb_ids = []
            metadatas = results.get('metadatas', [])
            
            # Handle both query and get result structures
            if isinstance(metadatas, list) and len(metadatas) > 0:
                # For query results, metadatas is a list of lists
                if isinstance(metadatas[0], list):
                    metadata_list = metadatas[0]
                else:
                    metadata_list = metadatas
                    
                for metadata in metadata_list:
                    if isinstance(metadata, dict):
                        mongodb_id = metadata.get('mongodb_id')
                        if mongodb_id:
                            try:
                                mongodb_ids.append(ObjectId(mongodb_id))
                            except Exception as e:
                                logger.warning(f"Invalid MongoDB ID {mongodb_id}: {e}")
            
            if not mongodb_ids:
                logger.warning("No valid MongoDB IDs found in ChromaDB results")
                return []
            
            # Build MongoDB query with date filtering
            mongo_filter = {"_id": {"$in": mongodb_ids}}
            
            # Add date range filter if specified
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                mongo_filter["document_date"] = date_filter
            
            # Retrieve full chunks from MongoDB
            mongo_results = list(self.collection.find(mongo_filter).limit(n_results))
            
            # Convert ObjectId to string for JSON serialization
            for chunk in mongo_results:
                chunk['_id'] = str(chunk['_id'])
            
            logger.info(f"Retrieved {len(mongo_results)} chunks matching criteria")
            return mongo_results
            
        except Exception as e:
            logger.error(f"Error in retrieve_chunks: {e}")
            logger.error(traceback.format_exc())
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            # Get ChromaDB stats
            chroma_count = self.chroma_collection.count()
            
            # Get MongoDB stats
            mongo_count = self.collection.count_documents({})
            
            # Get category breakdown from ChromaDB
            category_results = self.chroma_collection.get(include=['metadatas'])
            categories = {}
            companies = set()
            
            for metadata in category_results['metadatas']:
                category = metadata.get('category', 'Unknown')
                company = metadata.get('company_name', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
                companies.add(company)
            
            return {
                'chromadb_count': chroma_count,
                'mongodb_count': mongo_count,
                'companies': len(companies),
                'categories': categories,
                'company_list': sorted(list(companies))
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def update_incremental(self) -> Dict[str, int]:
        """Update ChromaDB with new chunks from MongoDB"""
        try:
            # Get all ChromaDB IDs
            existing_results = self.chroma_collection.get(include=['metadatas'])
            existing_mongodb_ids = set()
            
            for metadata in existing_results['metadatas']:
                mongodb_id = metadata.get('mongodb_id')
                if mongodb_id:
                    existing_mongodb_ids.add(mongodb_id)
            
            # Find new chunks in MongoDB
            all_chunks = list(self.collection.find({}))
            new_chunks = []
            
            for chunk in all_chunks:
                mongodb_id = str(chunk['_id'])
                if mongodb_id not in existing_mongodb_ids:
                    new_chunks.append(chunk)
            
            if not new_chunks:
                logger.info("No new chunks found for incremental update")
                return {'new_chunks': 0, 'embedded': 0, 'errors': 0}
            
            logger.info(f"Found {len(new_chunks)} new chunks for incremental update")
            
            # Process new chunks
            stats = {'new_chunks': len(new_chunks), 'embedded': 0, 'errors': 0}
            
            for chunk in new_chunks:
                try:
                    mongodb_id = str(chunk['_id'])
                    
                    # Extract text content
                    text_content = self._process_chunk_content(chunk)
                    if not text_content:
                        continue
                    
                    # Generate embedding
                    embedding = self._generate_embedding(text_content)
                    if embedding is None:
                        stats['errors'] += 1
                        continue
                    
                    # Add to ChromaDB
                    chunk_id = f"chunk_{mongodb_id}"
                    metadata = self._create_chunk_metadata(chunk)
                    
                    self.chroma_collection.add(
                        embeddings=[embedding],
                        metadatas=[metadata],
                        documents=[text_content[:1000]],
                        ids=[chunk_id]
                    )
                    
                    stats['embedded'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing new chunk {chunk.get('_id')}: {e}")
                    stats['errors'] += 1
            
            logger.info(f"Incremental update completed. Stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            return {'new_chunks': 0, 'embedded': 0, 'errors': 1}

    def search_similar_chunks(self, text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar chunks using semantic search"""
        return self.retrieve_chunks(query_text=text, n_results=n_results)

    def delete_collection(self):
        """Delete the ChromaDB collection (use with caution)"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def close(self):
        """Close database connections"""
        try:
            self.mongo_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

def main():
    """Example usage of AlphaSageVectorDB"""
    try:
        # Initialize vector database
        vector_db = AlphaSageVectorDB()
        
        # Get initial stats
        stats = vector_db.get_collection_stats()
        print(f"Initial stats: {stats}")
        
        # Embed chunks in batches (skip if already done)
        print("Starting batch embedding...")
        embedding_stats = vector_db.embed_chunks_batch(batch_size=50, skip_existing=True)
        print(f"Embedding completed: {embedding_stats}")
        
        # Get updated stats
        stats = vector_db.get_collection_stats()
        print(f"Updated stats: {stats}")
        
        # Example retrieval queries
        print("\n=== Example Queries ===")
        
        # Query 1: Future Insights for Ganesha Ecosphere (since that's what we have)
        results = vector_db.retrieve_chunks(
            company_name="Ganesha Ecosphere",
            category="Future Insights",
            n_results=3
        )
        print(f"Ganesha Ecosphere Future Insights: {len(results)} results")
        if results:
            print(f"Sample result: {results[0].get('content', {}).get('text', '')[:100]}...")
        
        # Query 2: Semantic search for revenue projections
        results = vector_db.retrieve_chunks(
            query_text="revenue projections growth financial performance",
            n_results=5
        )
        print(f"Revenue projections search: {len(results)} results")
        if results:
            print(f"Sample result: {results[0].get('content', {}).get('text', '')[:100]}...")
        
        # Query 3: Company Info for Ganesha Ecosphere
        results = vector_db.retrieve_chunks(
            company_name="Ganesha Ecosphere",
            category="Company Info",
            n_results=3
        )
        print(f"Ganesha Ecosphere Company Info: {len(results)} results")
        if results:
            print(f"Sample result: {results[0].get('content', {}).get('text', '')[:100]}...")
        
        # Query 4: Date range query
        results = vector_db.retrieve_chunks(
            company_name="Ganesha Ecosphere",
            start_date="2023-01-01",
            end_date="2025-12-31",
            n_results=3
        )
        print(f"Ganesha Ecosphere (2023-2025): {len(results)} results")
        
        # Query 5: Valuation Ratios
        results = vector_db.retrieve_chunks(
            company_name="Ganesha Ecosphere",
            category="Valuation Ratios",
            n_results=3
        )
        print(f"Ganesha Ecosphere Valuation Ratios: {len(results)} results")
        if results:
            print(f"Sample result: {results[0].get('content', {}).get('text', '')[:100]}...")
        
        # Close connections
        vector_db.close()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
