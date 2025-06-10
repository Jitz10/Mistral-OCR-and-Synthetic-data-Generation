from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import json
import asyncio
from dashboard_agent import DashboardAgent
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="alphasage.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize MongoDB connection
mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
db = mongo_client['alphasage']
dashboard_collection = db['dashboards']

BASE_UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

# Initialize DashboardAgent
dashboard_agent = DashboardAgent()

@app.route('/uploadReport', methods=['POST'])
def upload_report():
    company_name = request.form.get('companyName')
    company_ticker = request.form.get('companyTicker')

    file_fields = ['annualReports', 'fundReports', 'investorPresentations', 'concallTranscripts']
    uploaded_files = []

    for field in file_fields:
        field_folder = os.path.join(BASE_UPLOAD_FOLDER, field)
        os.makedirs(field_folder, exist_ok=True)

        files = request.files.getlist(field)
        for file in files:
            timestamp = int(time.time() * 1000)
            filename = f"{company_ticker}_{timestamp}_{secure_filename(file.filename)}"
            file.save(os.path.join(field_folder, filename))
        if files:
            uploaded_files.append(field)

    data_path = os.path.join(os.path.dirname(__file__), 'data.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception:
        return jsonify({'error': 'Failed to read or parse data.json'}), 500

    time.sleep(10)

    return jsonify({
        'message': 'Files uploaded and data processed successfully',
        'companyName': company_name,
        'companyTicker': company_ticker,
        'uploadedFiles': uploaded_files,
        'data': json_data
    })

@app.route('/generateDashboard', methods=['POST'])
async def generate_dashboard():
    """
    Generate dashboard JSON for a given company using yfinance data.
    Checks MongoDB for cached data first, generates new data if needed.
    """
    try:
        data = request.get_json()
        company_name = data.get('companyName')
        company_ticker = data.get('companyTicker')

        if not company_name or not company_ticker:
            return jsonify({
                'error': 'Both companyName and companyTicker are required'
            }), 400

        logger.info(f"Processing dashboard request for {company_name} ({company_ticker})")
        
        # Check MongoDB for cached data
        cached_data = dashboard_collection.find_one(
            {
                "company_name": company_name,
                "ticker": company_ticker,
                "created_at": {"$gte": datetime.now() - timedelta(days=2)}
            },
            sort=[("created_at", -1)]
        )

        if cached_data:
            logger.info(f"Returning cached dashboard for {company_name}")
            return jsonify({
                'message': 'Dashboard retrieved from cache',
                'companyName': company_name,
                'companyTicker': company_ticker,
                'data': cached_data['dashboard_data'],
                'cached': True,
                'cached_at': cached_data['created_at'].isoformat()
            })
        
        # Generate new dashboard
        logger.info(f"Generating new dashboard for {company_name}")
        dashboard_json = await dashboard_agent.generate_dashboard_json(company_name, company_ticker)
        
        return jsonify({
            'message': 'Dashboard generated successfully',
            'companyName': company_name,
            'companyTicker': company_ticker,
            'data': dashboard_json,
            'cached': False,
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
