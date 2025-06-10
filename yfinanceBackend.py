from flask import app,json,jsonify,Flask,request
from flask_cors import CORS
import yfinance as yf 
import os 
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import traceback
import requests
from datetime import datetime
from transformers import pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

@app.route("/stock_history",methods=['POST'])
def historic_data():
    try:
        data = request.get_json()
        company_name = data.get('company')
        duration = data.get('duration', '1mo')  # Default to 1 month

        if not company_name:
            return jsonify({'error': 'Company name is required'}), 400

        stock = yf.Ticker(company_name)
        #print(stock)
        print(stock.recommendations)
        history = stock.history(period=duration)

        if history.empty:
            return jsonify({'error': 'No data found for the given company'}), 404

        #print(history)
        print(history)
        history_data = history[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        history_data['Date'] = history_data['Date'].astype(str)

        history_data['Close_Open_%'] = ((history_data['Close'] - history_data['Open']) / history_data['Open']) * 100

        return jsonify({'data': history_data.to_dict(orient='records')})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/valuation_ratios", methods=["POST"])
def get_valuation_ratios():
    """Calculate comprehensive financial ratios and include OHLCV data."""
    try:
        body = request.get_json()
        ticker_symbol = body.get('company', '')
        duration = body.get('duration', '1y')

        ticker = yf.Ticker(ticker_symbol)
        hist_data = ticker.history(period=duration)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        current_info = ticker.info

        results = {
            "ticker": ticker_symbol,
            "duration": duration,
            "time_series_ratios": {},
            "average_ratios": {},
            "current_ratios": {}
        }

        for date, row in hist_data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            stock_price = row['Close']

            closest_financials = get_closest_financial_data(financials, date)
            closest_balance_sheet = get_closest_financial_data(balance_sheet, date)

            if closest_financials is not None and closest_balance_sheet is not None:
                eps = get_trailing_eps(ticker, date)
                revenue = closest_financials.get('Total Revenue', None)
                book_value_per_share = get_book_value_per_share(closest_balance_sheet, ticker, date)
                shares_outstanding = get_shares_outstanding(ticker, date)
                ebitda = closest_financials.get('EBITDA', None)

                market_cap = stock_price * shares_outstanding if shares_outstanding else None
                total_debt = get_total_debt(closest_balance_sheet)
                cash_and_cash_equiv = get_cash_and_equivalents(closest_balance_sheet)
                ev = (market_cap + total_debt - cash_and_cash_equiv) if all([market_cap, total_debt, cash_and_cash_equiv]) else None

                historical_data = {
                    "currentPrice": stock_price,
                    "trailingEps": eps,
                    "totalRevenue": revenue,
                    "bookValue": book_value_per_share,
                    "enterpriseValue": ev,
                    "ebitda": ebitda,
                    "sharesOutstanding": shares_outstanding,
                    "profitMargins": current_info.get("profitMargins"),
                    "grossMargins": current_info.get("grossMargins"),
                    "returnOnAssets": current_info.get("returnOnAssets"),
                    "returnOnEquity": current_info.get("returnOnEquity"),
                    "totalDebt": total_debt,
                    "marketCap": market_cap,
                    "totalAssets": closest_balance_sheet.get('Total Assets', None),
                    "currentRatio": current_info.get("currentRatio"),
                    "quickRatio": current_info.get("quickRatio"),
                    "dividendRate": current_info.get("dividendRate"),
                    "dividendYield": current_info.get("dividendYield"),
                    "payoutRatio": current_info.get("payoutRatio"),
                    "beta": current_info.get("beta"),
                    "shortRatio": current_info.get("shortRatio")
                }

                pe_ratio = stock_price / eps if eps and eps > 0 else None
                ps_ratio = (stock_price * shares_outstanding) / revenue if revenue and revenue > 0 and shares_outstanding else None
                pb_ratio = stock_price / book_value_per_share if book_value_per_share and book_value_per_share > 0 else None
                ev_revenue = ev / revenue if ev and revenue and revenue > 0 else None
                ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else None

                results["time_series_ratios"][date_str] = {
                    "P/E Ratio": pe_ratio,
                    "P/S Ratio": ps_ratio,
                    "P/B Ratio": pb_ratio,
                    "EV/Revenue": ev_revenue,
                    "EV/EBITDA": ev_ebitda,

                    # OHLCV
                    "Open": row['Open'],
                    "High": row['High'],
                    "Low": row['Low'],
                    "Close": row['Close'],
                    "Volume": row['Volume'],
                    "Stock Price": stock_price,

                    **get_profitability_ratios(historical_data),
                    **get_leverage_ratios(historical_data),
                    **get_liquidity_ratios(historical_data),
                    **get_dividend_ratios(historical_data),
                    **get_market_ratios(historical_data)
                }

        if results["time_series_ratios"]:
            all_ratios = {
                "P/E Ratio": [], "P/S Ratio": [], "P/B Ratio": [], "EV/Revenue": [], "EV/EBITDA": [],
                "Profit Margin": [], "Gross Margin": [], "Return on Assets (ROA)": [], "Return on Equity (ROE)": [],
                "Debt-to-Equity": [], "Debt-to-Assets": [],
                "Current Ratio": [], "Quick Ratio": [],
                "Dividend Rate": [], "Dividend Yield": [], "Payout Ratio": [],
                "Beta": [], "Short Ratio": []
            }

            for date_data in results["time_series_ratios"].values():
                for ratio_name, ratio_value in date_data.items():
                    if ratio_name in all_ratios and ratio_value is not None:
                        all_ratios[ratio_name].append(ratio_value)

            for ratio_name, ratio_values in all_ratios.items():
                results["average_ratios"][ratio_name] = sum(ratio_values) / len(ratio_values) if ratio_values else None

        # Current ratios
        data = current_info
        stock_price = data.get("currentPrice")
        eps = data.get("trailingEps")
        revenue = data.get("totalRevenue")
        book_value = data.get("bookValue")
        ev = data.get("enterpriseValue")
        ebitda = data.get("ebitda")
        shares_outstanding = data.get("sharesOutstanding")

        results["current_ratios"] = {
            "P/E Ratio": stock_price / eps if eps and eps > 0 else None,
            "P/S Ratio": stock_price / (revenue / shares_outstanding) if revenue and shares_outstanding and revenue > 0 else None,
            "P/B Ratio": stock_price / book_value if book_value and book_value > 0 else None,
            "EV/Revenue": ev / revenue if ev and revenue and revenue > 0 else None,
            "EV/EBITDA": ev / ebitda if ev and ebitda and ebitda > 0 else None,

            **get_profitability_ratios(data),
            **get_leverage_ratios(data),
            **get_liquidity_ratios(data),
            **get_dividend_ratios(data),
            **get_market_ratios(data)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
# Helper functions to retrieve and process financial data
def get_closest_financial_data(financials_df, target_date):
    """Get the closest available financial data to a target date."""
    if financials_df is None or financials_df.empty:
        return None
    
    # Convert target_date to datetime if it's not already
    if not isinstance(target_date, datetime):
        target_date = pd.to_datetime(target_date)
    
    # Handle timezone issues - convert both to timezone naive
    if hasattr(target_date, 'tz') and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Get the column dates
    dates = financials_df.columns
    
    # Convert all dates to tz-naive for comparison
    tz_naive_dates = []
    for date in dates:
        if hasattr(date, 'tz') and date.tz is not None:
            tz_naive_dates.append(date.tz_localize(None))
        else:
            tz_naive_dates.append(date)
    
    # Find the closest date
    if tz_naive_dates:
        closest_date_index = min(range(len(tz_naive_dates)), key=lambda i: abs(tz_naive_dates[i] - target_date))
        closest_date = dates[closest_date_index]  # Get the original date from the original list
    else:
        return None
    
    # Return the data for the closest date
    return financials_df[closest_date]

def get_trailing_eps(ticker, date):
    """Attempt to get the trailing EPS for a specific date."""
    try:
        # Try to get from earnings
        earnings = ticker.earnings
        if earnings is not None and not earnings.empty:
            # Handle timezone issues
            if hasattr(date, 'tz') and date.tz is not None:
                date_naive = date.tz_localize(None)
            else:
                date_naive = date
                
            # Get closest earnings date before the target date
            earnings_dates = earnings.index
            
            # Convert earnings dates to tz-naive for comparison
            valid_dates = []
            for d in earnings_dates:
                if hasattr(d, 'tz') and d.tz is not None:
                    d_naive = d.tz_localize(None)
                else:
                    d_naive = d
                if d_naive <= date_naive:
                    valid_dates.append((d, d_naive))
            
            if valid_dates:
                # Get the original date with the max naive date
                closest_date = max(valid_dates, key=lambda x: x[1])[0]
                return earnings.loc[closest_date, 'EPS']
        
        # Fallback to current info
        return ticker.info.get('trailingEps')
    except Exception as e:
        print(f"Error getting EPS: {e}")
        return None

def get_book_value_per_share(balance_sheet, ticker, date):
    """Calculate book value per share for a specific date."""
    try:
        total_assets = balance_sheet.get('Total Assets', None)
        total_liabilities = balance_sheet.get('Total Liabilities Net Minority Interest', None)
        
        if total_assets is not None and total_liabilities is not None:
            shareholders_equity = total_assets - total_liabilities
            shares_outstanding = get_shares_outstanding(ticker, date)
            
            if shares_outstanding:
                return shareholders_equity / shares_outstanding
        
        # Fallback to current info
        return ticker.info.get('bookValue')
    except:
        return None

def get_shares_outstanding(ticker, date):
    """Get the shares outstanding for a specific date."""
    try:
        # Try to get from historical data
        # Note: This is a simplification, as getting historical shares outstanding 
        # is complex and might require additional data sources
        return ticker.info.get('sharesOutstanding')
    except:
        return None

def get_total_debt(balance_sheet):
    """Get total debt from balance sheet."""
    try:
        return balance_sheet.get('Total Debt', None)
    except:
        return None

def get_cash_and_equivalents(balance_sheet):
    """Get cash and cash equivalents from balance sheet."""
    try:
        return balance_sheet.get('Cash And Cash Equivalents', None)
    except:
        return None

# Additional ratio functions

def get_profitability_ratios(data):
    """Calculate profitability ratios."""
    return {
        "Profit Margin": data.get("profitMargins"),
        "Gross Margin": data.get("grossMargins"),
        "Return on Assets (ROA)": data.get("returnOnAssets"),
        "Return on Equity (ROE)": data.get("returnOnEquity"),
    }

def get_leverage_ratios(data):
    """Calculate leverage ratios."""
    total_debt = data.get("totalDebt")
    market_cap = data.get("marketCap")
    total_assets = data.get("totalAssets")
    return {
        "Debt-to-Equity": total_debt / (market_cap - total_debt) if market_cap and total_debt else None,
        "Debt-to-Assets": total_debt / total_assets if total_assets and total_debt else None,
    }

def get_liquidity_ratios(data):
    """Calculate liquidity ratios."""
    return {
        "Current Ratio": data.get("currentRatio"),
        "Quick Ratio": data.get("quickRatio"),
    }

def get_dividend_ratios(data):
    """Calculate dividend ratios."""
    return {
        "Dividend Rate": data.get("dividendRate"),
        "Dividend Yield": data.get("dividendYield"),
        "Payout Ratio": data.get("payoutRatio"),
    }

def get_market_ratios(data):
    """Calculate market-related ratios."""
    return {
        "Beta": data.get("beta"),
        "Short Ratio": data.get("shortRatio"),
    }
    
@app.route("/detailed_comparison", methods=["POST"])
def detailed_comparison():
    """
    Compare detailed financial ratios for multiple companies over a specified duration.
    Takes a list of company ticker symbols and a duration parameter.
    Returns comprehensive comparative financial data in JSON format.
    """
    try:
        body = request.get_json()
        companies = body.get('companies', [])
        duration = body.get('duration', '1y')
        
        # Validate input
        if not companies:
            return jsonify({"error": "No companies provided"}), 400
            
        # Initialize results container
        comparison_results = {
            "duration": duration,
            "companies": {},
            "comparison_summary": {}
        }
        
        # For each company, calculate the valuation ratios
        for company in companies:
            # Create a mock request body for the get_valuation_ratios function
            mock_request_body = {
                'company': company,
                'duration': duration
            }
            
            # Get valuation ratios by leveraging the existing function
            # We'll simulate the request context to reuse the function
            with app.test_request_context(json=mock_request_body):
                # Call the function directly rather than through the route
                valuation_response = get_valuation_ratios()
                
                # Convert response to dict if it's a Response object
                if isinstance(valuation_response, tuple):
                    # Error occurred in the function
                    return jsonify({"error": f"Error processing {company}: {valuation_response[0].json}"}), 500
                
                if hasattr(valuation_response, 'json'):
                    valuation_data = valuation_response.json
                else:
                    valuation_data = json.loads(valuation_response.data)
                
                # Store the results for this company
                comparison_results["companies"][company] = valuation_data
        
        # Generate comparison summary for current ratios
        ratio_categories = [
            # Valuation ratios
            "P/E Ratio", "P/S Ratio", "P/B Ratio", "EV/Revenue", "EV/EBITDA",
            # Profitability ratios
            "Profit Margin", "Gross Margin", "Return on Assets (ROA)", "Return on Equity (ROE)",
            # Leverage ratios
            "Debt-to-Equity", "Debt-to-Assets",
            # Liquidity ratios
            "Current Ratio", "Quick Ratio",
            # Dividend ratios
            "Dividend Rate", "Dividend Yield", "Payout Ratio",
            # Market ratios
            "Beta", "Short Ratio"
        ]
        
        # For each ratio, find the min, max, and average across companies
        for ratio in ratio_categories:
            ratio_values = []
            company_ratios = {}
            
            for company, data in comparison_results["companies"].items():
                if "current_ratios" in data and ratio in data["current_ratios"]:
                    ratio_value = data["current_ratios"][ratio]
                    if ratio_value is not None:
                        ratio_values.append(ratio_value)
                        company_ratios[company] = ratio_value
            
            if ratio_values:
                comparison_results["comparison_summary"][ratio] = {
                    "min": min(ratio_values),
                    "max": max(ratio_values),
                    "avg": sum(ratio_values) / len(ratio_values),
                    "by_company": company_ratios,
                    # Find companies with min and max values
                    "min_company": min(company_ratios.items(), key=lambda x: x[1])[0] if company_ratios else None,
                    "max_company": max(company_ratios.items(), key=lambda x: x[1])[0] if company_ratios else None
                }
        
        # Add time series comparison for stock price trends
        comparison_results["price_trends"] = {}
        
        # Collect all dates from all companies
        all_dates = set()
        for company, data in comparison_results["companies"].items():
            if "time_series_ratios" in data:
                all_dates.update(data["time_series_ratios"].keys())
        
        # For each date, collect stock prices from all companies
        for date in sorted(all_dates):
            comparison_results["price_trends"][date] = {}
            for company, data in comparison_results["companies"].items():
                if "time_series_ratios" in data and date in data["time_series_ratios"]:
                    stock_price = data["time_series_ratios"][date].get("Stock Price")
                    if stock_price is not None:
                        comparison_results["price_trends"][date][company] = stock_price
        
        return jsonify(comparison_results)
    
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/top_companies_sector',methods=["POST"])
def top_industrials():
    body = request.get_json()
    industry = body.get("sector")
    data = yf.Sector(key=industry)
    companies_df = data.top_companies[:10].reset_index()  # symbol becomes a column
    companies_df = companies_df.rename(columns={"market weight": "market_weight"})
    # Convert to list of dicts
    companies = json.loads(companies_df.to_json(orient='records'))
    #print(companies)
    enriched_data = []

    for company in companies:
        symbol = company['symbol']
        ticker_info = yf.Ticker(symbol).info

        current_price = ticker_info.get('currentPrice')
        previous_close = ticker_info.get('previousClose')
        pe_ratio = ticker_info.get('trailingPE')

        percent_change = None
        if current_price and previous_close:
            try:
                percent_change = round(((current_price - previous_close) / previous_close) * 100, 2)
            except:
                percent_change = None

        company.update({
            "stock_price": current_price,
            "percent_change": percent_change,
            "pe_ratio": pe_ratio
        })

        enriched_data.append(company)

    return jsonify(enriched_data)

@app.route('/sector_info', methods=['POST'])
def get_sector_info():
    try:
        # Parse the sector key from the JSON body
        data = request.get_json()
        sector_key = data.get('sector')

        if not sector_key:
            return jsonify({"error": "Sector 'key' is required in JSON body"}), 400

        # Get sector info using yfinance
        sector_data = yf.Sector(key=sector_key)
        ticker_info = sector_data.ticker.info

        return jsonify(ticker_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


NEWS_API_KEY = os.getenv('NEWS_API_KEY')



@app.route('/api/stock-market-headlines', methods=['GET'])
def get_stock_market_headlines():
    try:
        # Get top headlines specifically about business/finance
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': NEWS_API_KEY,
            'category': 'business',
            'language': 'en',
            'pageSize': 10
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        print(data)
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': f"API Error: {data.get('message', 'Unknown error')}",
                'code': response.status_code
            }), response.status_code
        
        # Import sentiment analysis components
        from transformers import pipeline
        
        # Load a lightweight sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False
        )
        
        # Extract headlines with sentiment analysis
        headlines = []
        for article in data.get('articles', []):
            # Determine text for sentiment analysis
            text_for_sentiment = article.get('description')
            if text_for_sentiment is None or text_for_sentiment == "":
                text_for_sentiment = article.get('title')
            
            # Perform sentiment analysis
            sentiment = "neutral"  # Default
            if text_for_sentiment:
                try:
                    result = sentiment_analyzer(text_for_sentiment)[0]
                    # Map the result to simple positive/negative/neutral categories
                    label = result['label'].lower()
                    if 'positive' in label:
                        sentiment = "positive"
                    elif 'negative' in label:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                except Exception as e:
                    print(f"Sentiment analysis failed: {e}")
            
            headlines.append({
                'title': article.get('title'),
                'source': article.get('source', {}).get('name'),
                'description': article.get('description'),
                'image': article.get('urlToImage'),
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'sentiment': sentiment
            })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'count': len(headlines),
            'headlines': headlines
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)