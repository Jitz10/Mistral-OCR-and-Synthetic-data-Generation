import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import traceback
from typing import Dict, Any, Optional
import autogen
import warnings
warnings.filterwarnings('ignore')

# Today's date context
TODAY = datetime.now()  # Use actual current date instead of hardcoded
TODAY_STR = TODAY.strftime("%Y-%m-%d")

# Configuration - consider using environment variables for API keys
config_list = [
    {
        "model": "llama3-70b-8192",
        "api_key": "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",  # Replace with actual key or use env var
        "api_type": "groq"
    }
]

def get_financial_ratio_by_date(company_ticker: str, ratio_name: str, target_date: str, duration: str = '2y') -> Dict[str, Any]:
    """
    Get financial ratio for a company as of a specific date.
    
    Args:
        company_ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        ratio_name (str): Name of the financial ratio to retrieve
        target_date (str): Target date in YYYY-MM-DD format
        duration (str): Time period to fetch data around the target date ('1y', '2y', '5y', etc.)
    
    Returns:
        dict: Contains ratio value as of the target date, closest available date, and context data
    """
    
    try:
        # Validate inputs
        if not company_ticker:
            return {"error": "Company ticker is required"}
        
        if not ratio_name:
            return {"error": "Ratio name is required"}
            
        if not target_date:
            return {"error": "Target date is required"}
        
        # Parse target date
        try:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD (e.g., '2025-12-31')"}
        
        # Check if target date is not in the future
        if target_dt > TODAY:
            return {"error": f"Target date cannot be in the future. Today is {TODAY_STR}"}
        
        # Calculate days from today for context
        days_from_today = (TODAY - target_dt).days
        
        # Available ratios
        available_ratios = {
            'P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'EV/Revenue', 'EV/EBITDA',
            'Profit Margin', 'Gross Margin', 'Return on Assets (ROA)', 'Return on Equity (ROE)',
            'Debt-to-Equity', 'Debt-to-Assets', 'Current Ratio', 'Quick Ratio',
            'Dividend Rate', 'Dividend Yield', 'Payout Ratio', 'Beta', 'Short Ratio',
            'Open', 'High', 'Low', 'Close', 'Volume', 'Stock Price'
        }
        
        if ratio_name not in available_ratios:
            return {
                "error": f"Ratio '{ratio_name}' not available",
                "available_ratios": sorted(list(available_ratios))
            }
        
        # Calculate date range around target date
        end_date = min(target_dt + timedelta(days=30), TODAY)  # Don't exceed today
        
        # Parse duration more robustly
        if duration.endswith('y'):
            years = int(duration[:-1])
            start_date = target_dt - timedelta(days=years * 365)
        elif duration.endswith('m'):
            months = int(duration[:-1])
            start_date = target_dt - timedelta(days=months * 30)
        elif duration.endswith('d'):
            days = int(duration[:-1])
            start_date = target_dt - timedelta(days=days)
        else:
            start_date = target_dt - timedelta(days=365)  # Default to 1 year
        
        # Get data with error handling
        try:
            ticker = yf.Ticker(company_ticker.upper())
            hist_data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=False)
        except Exception as e:
            return {"error": f"Failed to fetch data for {company_ticker}: {str(e)}"}
        
        if hist_data.empty:
            return {"error": f"No historical data found for {company_ticker} around {target_date}"}
        
        # Find closest date to target
        hist_data.index = pd.to_datetime(hist_data.index)
        
        # Handle timezone issues more robustly
        if hist_data.index.tz is not None:
            target_dt_tz = target_dt.replace(tzinfo=hist_data.index.tz)
        else:
            target_dt_tz = target_dt
        
        # Find the closest date
        date_diffs = abs(hist_data.index - target_dt_tz)
        closest_date_idx = date_diffs.argmin()
        closest_date = hist_data.index[closest_date_idx]
        closest_row = hist_data.iloc[closest_date_idx]
        
        # Get company info with better error handling
        try:
            current_info = ticker.info
            if not current_info or len(current_info) < 5:  # Basic check for valid info
                current_info = {}
        except:
            current_info = {}
        
        # Get financial statements with better error handling
        financials = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        cashflow = pd.DataFrame()
        
        try:
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
        except Exception as e:
            print(f"Warning: Could not fetch financial statements: {e}")
        
        # Calculate the requested ratio
        ratio_value = None
        stock_price = closest_row['Close']
        
        # Enhanced ratio calculations with better error handling
        if ratio_name == 'P/E Ratio':
            eps = current_info.get('trailingEps') or current_info.get('forwardEps')
            ratio_value = stock_price / eps if eps and eps > 0 else None
            
        elif ratio_name == 'P/S Ratio':
            revenue = current_info.get('totalRevenue')
            shares = current_info.get('sharesOutstanding')
            if revenue and shares and revenue > 0:
                ratio_value = stock_price / (revenue / shares)
                
        elif ratio_name == 'P/B Ratio':
            book_value = current_info.get('bookValue')
            ratio_value = stock_price / book_value if book_value and book_value > 0 else None
            
        elif ratio_name in ['Profit Margin', 'Gross Margin', 'Return on Assets (ROA)', 'Return on Equity (ROE)',
                           'Current Ratio', 'Quick Ratio', 'Dividend Rate', 'Dividend Yield', 'Payout Ratio',
                           'Beta', 'Short Ratio']:
            # Direct mapping for simple ratios
            ratio_mapping = {
                'Profit Margin': 'profitMargins',
                'Gross Margin': 'grossMargins',
                'Return on Assets (ROA)': 'returnOnAssets',
                'Return on Equity (ROE)': 'returnOnEquity',
                'Current Ratio': 'currentRatio',
                'Quick Ratio': 'quickRatio',
                'Dividend Rate': 'dividendRate',
                'Dividend Yield': 'dividendYield',
                'Payout Ratio': 'payoutRatio',
                'Beta': 'beta',
                'Short Ratio': 'shortRatio'
            }
            ratio_value = current_info.get(ratio_mapping.get(ratio_name))
            
        elif ratio_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
            ratio_value = closest_row[ratio_name]
            
        elif ratio_name == 'Stock Price':
            ratio_value = stock_price
            
        # Calculate days difference
        if hasattr(closest_date, 'tz') and closest_date.tz:
            closest_date_naive = closest_date.tz_localize(None)
        else:
            closest_date_naive = closest_date
            
        days_difference = abs((closest_date_naive - target_dt).days)
        
        # Get context data more efficiently
        context_data = {}
        closest_idx = hist_data.index.get_loc(closest_date)
        
        # Get 5 data points before and after (if available)
        start_idx = max(0, closest_idx - 5)
        end_idx = min(len(hist_data), closest_idx + 6)
        context_range = hist_data.iloc[start_idx:end_idx]
        
        for date, row in context_range.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            if ratio_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                context_data[date_str] = float(row[ratio_name])
            elif ratio_name == 'Stock Price':
                context_data[date_str] = float(row['Close'])
        
        # Enhanced result with better metadata
        result = {
            "ticker": company_ticker.upper(),
            "company_name": current_info.get('longName', 'N/A'),
            "ratio_name": ratio_name,
            "target_date": target_date,
            "closest_available_date": closest_date.strftime('%Y-%m-%d'),
            "days_difference": days_difference,
            "days_from_today": days_from_today,
            "today": TODAY_STR,
            "ratio_value": float(ratio_value) if ratio_value is not None else None,
            "stock_price_on_date": float(stock_price),
            "context_data": context_data,
            "date_context": {
                "is_recent": days_from_today <= 30,
                "is_current_year": target_dt.year == TODAY.year,
                "is_weekend_request": target_dt.weekday() >= 5,
                "market_status": "Market closed (weekend)" if target_dt.weekday() >= 5 else "Market trading day"
            },
            "data_quality": {
                "has_historical_price": True,
                "has_company_info": bool(current_info),
                "has_historical_financials": not financials.empty,
                "has_historical_balance_sheet": not balance_sheet.empty,
                "data_freshness": "good" if days_difference <= 7 else "moderate" if days_difference <= 30 else "stale"
            },
            "note": f"Data retrieved for {company_ticker.upper()} as of {closest_date.strftime('%Y-%m-%d')} "
                   f"({days_difference} days from target date, {days_from_today} days ago from today)"
        }
        
        return result
        
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "traceback": traceback.format_exc()[-1000:]  # Limit traceback length
        }

def validate_ticker(ticker: str) -> bool:
    """Validate if a ticker symbol exists"""
    try:
        test_ticker = yf.Ticker(ticker.upper())
        info = test_ticker.info
        return bool(info and info.get('symbol'))
    except:
        return False

# Enhanced agent with better error handling and validation
financial_agent = autogen.ConversableAgent(
    name="FinancialAgent",
    llm_config={
        "config_list": config_list,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_financial_ratio_by_date",
                    "description": "Get financial ratio for a company as of a specific date. Returns detailed financial metrics with context data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')"
                            },
                            "ratio_name": {
                                "type": "string",
                                "description": "Financial ratio name. Options: 'P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'Stock Price', 'ROE', 'Current Ratio', 'Beta', 'Open', 'High', 'Low', 'Close', 'Volume', etc."
                            },
                            "target_date": {
                                "type": "string",
                                "description": "Target date in YYYY-MM-DD format (e.g., '2024-12-31'). Cannot be in the future."
                            },
                            "duration": {
                                "type": "string",
                                "description": "Time period for context data ('1y', '2y', '5y', '6m', '90d')",
                                "default": "2y"
                            }
                        },
                        "required": ["company_ticker", "ratio_name", "target_date"]
                    }
                }
            }
        ]
    },
    system_message=f"""You are an expert financial analyst assistant. Today is {TODAY_STR}.

Key capabilities:
- Retrieve historical stock data and financial ratios for specific dates
- Provide detailed analysis with context and data quality indicators
- Handle various financial metrics including valuation, profitability, and market ratios

When helping users:
1. Always validate ticker symbols and dates
2. Explain what the financial ratio means in simple terms
3. Provide context about data quality and availability
4. Suggest related metrics that might be useful
5. Handle errors gracefully with helpful suggestions

Available ratios include:
- Valuation: P/E Ratio, P/S Ratio, P/B Ratio, EV/Revenue, EV/EBITDA
- Profitability: Profit Margin, Gross Margin, ROE, ROA
- Liquidity: Current Ratio, Quick Ratio
- Market: Stock Price, Beta, Volume
- Price Data: Open, High, Low, Close

Remember: Some fundamental ratios may use current company data as historical fundamental data varies in availability."""
)

# Register function with better error handling
@financial_agent.register_for_execution()
@financial_agent.register_for_llm(name="get_financial_ratio_by_date", description="Get financial ratio for a specific date")
def get_financial_ratio_by_date_wrapper(company_ticker: str, ratio_name: str, target_date: str, duration: str = '2y') -> Dict[str, Any]:
    return get_financial_ratio_by_date(company_ticker, ratio_name, target_date, duration)

# Enhanced user proxy with better termination handling
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").lower().strip() in ["exit", "quit", "bye", "goodbye"],
    code_execution_config=False,
    max_consecutive_auto_reply=10,
)

# Example usage function
def run_financial_analysis():
    """Run the financial analysis chat interface"""
    print(f"🏦 Financial Analysis Assistant - {TODAY_STR}")
    print("=" * 50)
    print("Available commands:")
    print("• Get financial ratios for specific dates")
    print("• Analyze stock performance")
    print("• Compare financial metrics")
    print("\nExamples:")
    print("- 'What was AAPL P/E ratio on 2024-01-15?'")
    print("- 'Get TSLA stock price for 2023-12-31'")
    print("- 'Show me MSFT ROE as of 2024-06-30'")
    print("\nType 'exit' to quit\n")
    
    user_proxy.initiate_chat(
        financial_agent,
        message="Hello! I'm ready to help you analyze financial data. What would you like to know?"
    )

if __name__ == "__main__":
    run_financial_analysis()