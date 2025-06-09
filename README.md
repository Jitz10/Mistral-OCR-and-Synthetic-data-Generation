# AlphaSage Financial Analysis Tools

This repository contains tools for analyzing Indian equities using AutoGen Micro and Macro Agents.

## Setup

### Prerequisites

- Python 3.8+
- Docker Desktop (for Redis)
- MongoDB (local or remote)

### Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying the sample file:
   ```
   copy alphasage.env .env
   ```

5. Set up Redis using the provided script:
   ```
   setup_redis.bat
   ```

## Redis Setup

The tools use Redis for efficient caching. You have several options to set up Redis:

### Option 1: Using Docker (Recommended)

Run the `setup_redis.bat` script to automatically set up Redis in Docker.

Alternatively, manually run:
```
docker run --name alphasage-redis -p 6379:6379 -d redis:latest
```

### Option 2: Using Windows Subsystem for Linux (WSL)

If you have WSL installed:

```bash
# In WSL terminal
sudo apt update
sudo apt install redis-server

# Start Redis
sudo service redis-server start

# Test Redis
redis-cli ping
```

### Option 3: Using Memurai (Windows Native Redis Alternative)

1. Download Memurai from: https://www.memurai.com/
2. Install and run it
3. It will run on port 6379 by default

### Option 4: Memory Cache Fallback

The system will automatically fall back to in-memory caching if Redis is not available.

## Running the Tools

```
python tools.py
```

## Files

- `tools.py` - Core financial analysis tools
- `vector.py` - Vector database tools for RAG search
- `OCR2.py` - OCR processing for financial documents
- `setup_redis.bat` - Utility script to set up Redis
