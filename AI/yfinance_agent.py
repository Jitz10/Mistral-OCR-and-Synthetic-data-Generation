import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import traceback
from typing import Dict, Any, Optional, List
import autogen
import warnings
import os
warnings.filterwarnings('ignore')

# Today's date context
TODAY = datetime.now()
TODAY_STR = TODAY.strftime("%Y-%m-%d")

# Configuration - Use environment variables for API keys
config_list = [
    {
        "model": "llama3-70b-8192",
        "api_key": "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", # Use environment variable
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
            'Open', 'High', 'Low', 'Close', 'Volume', 'Stock Price',
            'Operating Income % Sales', 'Depreciation % Sales', 'Sales Expenses % Sales',
            'CFO/Sales', 'CFO/Total Assets', 'CFO/Total Debt'
        }
        
        if ratio_name not in available_ratios:
            return {
                "error": f"Ratio '{ratio_name}' not available",
                "available_ratios": sorted(list(available_ratios))
            }
        
        # Calculate date range around target date
        end_date = min(target_dt + timedelta(days=30), TODAY)
        
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
            start_date = target_dt - timedelta(days=365)
        
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
        
        # Handle timezone issues
        if hist_data.index.tz is not None:
            target_dt_tz = target_dt.replace(tzinfo=hist_data.index.tz)
        else:
            target_dt_tz = target_dt
        
        # Find the closest date
        date_diffs = abs(hist_data.index - target_dt_tz)
        closest_date_idx = date_diffs.argmin()
        closest_date = hist_data.index[closest_date_idx]
        closest_row = hist_data.iloc[closest_date_idx]
        
        # Get company info with error handling
        try:
            current_info = ticker.info
            if not current_info or len(current_info) < 5:
                current_info = {}
        except:
            current_info = {}
        
        # Get financial statements
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
        
        # Enhanced ratio calculations
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
            
        elif ratio_name == 'Operating Income % Sales':
            try:
                op_income = financials.loc['Operating Income'].iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                ratio_value = op_income / revenue if revenue else None
            except:
                ratio_value = None
                
        elif ratio_name == 'Depreciation % Sales':
            try:
                depreciation = cashflow.loc['Depreciation'].iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                ratio_value = depreciation / revenue if revenue else None
            except:
                ratio_value = None
                
        elif ratio_name == 'Sales Expenses % Sales':
            try:
                sga = financials.loc.get('Selling General and Administrative', pd.Series([None])).iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                ratio_value = sga / revenue if revenue else None
            except:
                ratio_value = None
                
        elif ratio_name == 'CFO/Sales':
            try:
                cfo = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                ratio_value = cfo / revenue if revenue else None
            except:
                ratio_value = None
                
        elif ratio_name == 'CFO/Total Assets':
            try:
                cfo = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                ratio_value = cfo / total_assets if total_assets else None
            except:
                ratio_value = None
                
        elif ratio_name == 'CFO/Total Debt':
            try:
                cfo = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                short_term_debt = balance_sheet.loc.get('Short Term Debt', pd.Series([0])).iloc[0]
                long_term_debt = balance_sheet.loc.get('Long Term Debt', pd.Series([0])).iloc[0]
                total_debt = short_term_debt + long_term_debt
                ratio_value = cfo / total_debt if total_debt else None
            except:
                ratio_value = None
        
        # Calculate days difference
        if hasattr(closest_date, 'tz') and closest_date.tz:
            closest_date_naive = closest_date.tz_localize(None)
        else:
            closest_date_naive = closest_date
            
        days_difference = abs((closest_date_naive - target_dt).days)
        
        # Get context data
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
        
        # Enhanced result
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
            "traceback": traceback.format_exc()[-1000:]
        }

def validate_ticker(ticker: str) -> bool:
    """Validate if a ticker symbol exists"""
    try:
        test_ticker = yf.Ticker(ticker.upper())
        info = test_ticker.info
        return bool(info and info.get('symbol'))
    except:
        return False

def create_financial_ratios_csv(
    company_ticker: str,
    ratio_names: List[str],
    start_date: str,
    end_date: str,
    filename: Optional[str] = None,
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Create a CSV file with time-series financial ratios and statistics.
    
    Args:
        company_ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        ratio_names (List[str]): List of financial ratios to include
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        filename (str, optional): Output filename (defaults to {ticker}_ratios.csv)
        output_dir (str): Output directory path
    
    Returns:
        dict: Contains success status, filename, statistics, and any errors
    """
    
    try:
        # Input validation
        if not company_ticker:
            return {"error": "Company ticker is required"}
        
        if not ratio_names or len(ratio_names) == 0:
            return {"error": "At least one ratio name is required"}
            
        if not start_date or not end_date:
            return {"error": "Both start_date and end_date are required"}
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        
        # Validate date range
        if start_dt >= end_dt:
            return {"error": "Start date must be before end date"}
            
        if end_dt > TODAY:
            return {"error": f"End date cannot be in the future. Today is {TODAY_STR}"}
        
        # Available ratios
        available_ratios = {
            'Close', 'Volume', 'P/E Ratio', 'P/S Ratio', 'P/B Ratio',
            'Profit Margin', 'Gross Margin', 'ROE', 'ROA', 'Current Ratio',
            'Quick Ratio', 'Debt-to-Equity', 'Dividend Yield', 'Beta'
        }
        
        # Validate ratio names
        invalid_ratios = [r for r in ratio_names if r not in available_ratios]
        if invalid_ratios:
            return {
                "error": f"Invalid ratios: {invalid_ratios}",
                "available_ratios": sorted(list(available_ratios))
            }
        
        # Fetch ticker data
        try:
            ticker = yf.Ticker(company_ticker.upper())
            hist_data = ticker.history(start=start_dt, end=end_dt + timedelta(days=1), auto_adjust=True)
            
            if hist_data.empty:
                return {"error": f"No historical data found for {company_ticker} in the specified date range"}
                
        except Exception as e:
            return {"error": f"Failed to fetch data for {company_ticker}: {str(e)}"}
        
        # Get company info for fundamental ratios
        try:
            company_info = ticker.info
            company_name = company_info.get('longName', company_ticker.upper())
        except:
            company_info = {}
            company_name = company_ticker.upper()
        
        # Prepare the main dataframe
        hist_data.index = pd.to_datetime(hist_data.index)
        result_df = pd.DataFrame(index=hist_data.index)
        result_df['Date'] = hist_data.index.strftime('%Y-%m-%d')
        
        # Calculate each requested ratio
        for ratio_name in ratio_names:
            if ratio_name == 'Close':
                result_df[ratio_name] = hist_data['Close'].round(4)
                
            elif ratio_name == 'Volume':
                result_df[ratio_name] = hist_data['Volume'].astype(int)
                
            elif ratio_name == 'P/E Ratio':
                eps = company_info.get('trailingEps') or company_info.get('forwardEps')
                if eps and eps > 0:
                    result_df[ratio_name] = (hist_data['Close'] / eps).round(4)
                else:
                    result_df[ratio_name] = None
                    
            elif ratio_name == 'P/S Ratio':
                revenue = company_info.get('totalRevenue')
                shares = company_info.get('sharesOutstanding')
                if revenue and shares and revenue > 0:
                    revenue_per_share = revenue / shares
                    result_df[ratio_name] = (hist_data['Close'] / revenue_per_share).round(4)
                else:
                    result_df[ratio_name] = None
                    
            elif ratio_name == 'P/B Ratio':
                book_value = company_info.get('bookValue')
                if book_value and book_value > 0:
                    result_df[ratio_name] = (hist_data['Close'] / book_value).round(4)
                else:
                    result_df[ratio_name] = None
                    
            else:
                # For other fundamental ratios, use current values
                ratio_mapping = {
                    'Profit Margin': 'profitMargins',
                    'Gross Margin': 'grossMargins',
                    'ROE': 'returnOnEquity',
                    'ROA': 'returnOnAssets',
                    'Current Ratio': 'currentRatio',
                    'Quick Ratio': 'quickRatio',
                    'Debt-to-Equity': 'debtToEquity',
                    'Dividend Yield': 'dividendYield',
                    'Beta': 'beta'
                }
                
                ratio_value = company_info.get(ratio_mapping.get(ratio_name))
                if ratio_value is not None:
                    result_df[ratio_name] = round(float(ratio_value), 4)
                else:
                    result_df[ratio_name] = None
        
        # Reset index to make Date a regular column
        result_df = result_df.reset_index(drop=True)
        
        # Set filename
        if filename is None:
            filename = f"{company_ticker.upper()}_ratios.csv"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        result_df.to_csv(filepath, index=False)
        
        # Calculate statistics for numeric columns
        stats = {}
        numeric_columns = result_df.select_dtypes(include=[float, int]).columns
        
        for col in numeric_columns:
            if col != 'Volume':
                col_data = result_df[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        'count': len(col_data),
                        'mean': round(col_data.mean(), 4),
                        'std': round(col_data.std(), 4),
                        'min': round(col_data.min(), 4),
                        'max': round(col_data.max(), 4),
                        'median': round(col_data.median(), 4)
                    }
        
        # Summary statistics
        total_days = len(result_df)
        trading_days = len(result_df[result_df['Close'].notna()])
        
        return {
            "success": True,
            "ticker": company_ticker.upper(),
            "company_name": company_name,
            "filename": filename,
            "filepath": filepath,
            "date_range": {
                "start": start_date,
                "end": end_date,
                "total_days": total_days,
                "trading_days": trading_days
            },
            "ratios_included": ratio_names,
            "statistics": stats,
            "data_shape": {
                "rows": len(result_df),
                "columns": len(result_df.columns)
            },
            "preview": result_df.head(5).to_dict('records'),
            "note": f"CSV file created successfully at {filepath} with {total_days} data points for {company_ticker.upper()}"
        }
        
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "traceback": traceback.format_exc()[-1000:]
        }

# Create AutoGen agents
def create_financial_agent():
    """Create and configure the financial agent"""
    
    financial_agent = autogen.ConversableAgent(
        name="FinancialAgent",
        llm_config={
            "config_list": config_list,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_financial_ratios_csv",
                        "description": "Create a CSV file with time-series financial ratios and statistics for a given date range.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "company_ticker": {
                                    "type": "string",
                                    "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')"
                                },
                                "ratio_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of financial ratios. Options: 'Close', 'Volume', 'P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'Profit Margin', 'Gross Margin', 'ROE', 'ROA', 'Current Ratio', 'Quick Ratio', 'Debt-to-Equity', 'Dividend Yield', 'Beta'"
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "Start date in YYYY-MM-DD format (e.g., '2023-01-01')"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "End date in YYYY-MM-DD format (e.g., '2024-12-31'). Cannot be in the future."
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "Output filename (optional, defaults to {ticker}_ratios.csv)",
                                    "default": None
                                },
                                "output_dir": {
                                    "type": "string",
                                    "description": "Output directory path (optional, defaults to current directory)",
                                    "default": "."
                                }
                            },
                            "required": ["company_ticker", "ratio_names", "start_date", "end_date"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_financial_ratio_by_date",
                        "description": "Get financial ratio for a company as of a specific date.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "company_ticker": {
                                    "type": "string",
                                    "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')"
                                },
                                "ratio_name": {
                                    "type": "string",
                                    "description": "Financial ratio name"
                                },
                                "target_date": {
                                    "type": "string",
                                    "description": "Target date in YYYY-MM-DD format"
                                },
                                "duration": {
                                    "type": "string",
                                    "description": "Time period for context data",
                                    "default": "2y"
                                }
                            },
                            "required": ["company_ticker", "ratio_name", "target_date"]
                        }
                    }
                }
            ]
        },
        system_message=f"""You are an expert financial analyst assistant with comprehensive data analysis capabilities. Today is {TODAY_STR}.

Key capabilities:
1. CSV Generation: Create time-series data files with multiple ratios
2. Single-point Analysis: Get specific ratio values for target dates
3. Statistical Analysis: Provide detailed statistics and insights

Available ratios for analysis:
- Price Data: Close, Volume, Open, High, Low, Stock Price
- Valuation: P/E Ratio, P/S Ratio, P/B Ratio, EV/Revenue, EV/EBITDA
- Profitability: Profit Margin, Gross Margin, ROE, ROA, Operating Income % Sales
- Financial Health: Current Ratio, Quick Ratio, Debt-to-Equity, Debt-to-Assets
- Market Metrics: Dividend Rate, Dividend Yield, Payout Ratio, Beta, Short Ratio
- Cash Flow: CFO/Sales, CFO/Total Assets, CFO/Total Debt
- Operating Metrics: Depreciation % Sales, Sales Expenses % Sales

When creating CSVs:
- Focus on essential metrics only
- Include statistical summaries
- Provide data quality indicators
- Create organized output structure with proper file naming

Best practices:
- Validate all inputs thoroughly
- Handle missing data gracefully
- Use environment variables for API keys
- Provide comprehensive error messages
- Include data quality assessments"""
    )
    
    # Register functions
    @financial_agent.register_for_execution()
    def create_csv_wrapper(company_ticker: str, ratio_names: List[str], start_date: str, end_date: str, filename: Optional[str] = None, output_dir: str = ".") -> Dict[str, Any]:
        return create_financial_ratios_csv(company_ticker, ratio_names, start_date, end_date, filename, output_dir)

    @financial_agent.register_for_execution()
    def get_ratio_wrapper(company_ticker: str, ratio_name: str, target_date: str, duration: str = '2y') -> Dict[str, Any]:
        return get_financial_ratio_by_date(company_ticker, ratio_name, target_date, duration)
    
    return financial_agent

def run_financial_analysis():
    """Run the enhanced financial analysis chat interface"""
    
    # Create agents
    financial_agent = create_financial_agent()
    
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content").lower().strip() in ["exit", "quit", "bye", "goodbye"],
        code_execution_config=False,
        max_consecutive_auto_reply=10,
    )
    
    print(f"🏦 Enhanced Financial Analysis Assistant - {TODAY_STR}")
    print("=" * 60)
    print("🆕 CSV Generation & Single-Point Analysis Capabilities!")
    print("\nAvailable commands:")
    print("• Generate CSV files with time-series ratios")
    print("• Get single-point financial ratios")
    print("• Analyze trends and statistics")
    print("\nCSV Examples:")
    print("- 'Create CSV for AAPL with Close, P/E Ratio, ROE from 2023-01-01 to 2024-12-31'")
    print("- 'Generate TSLA ratios CSV: Close, Volume, Beta for last 2 years'")
    print("- 'Make CSV for MSFT profitability ratios 2024 data'")
    print("\nSingle-point Examples:")
    print("- 'What was AAPL P/E ratio on 2024-01-15?'")
    print("- 'Get TSLA closing price for 2023-12-31'")
    print("\nSecurity Note: Set GROQ_API_KEY environment variable")
    print("Type 'exit' to quit\n")
    
    # Start the chat with a welcome message
    user_proxy.initiate_chat(
        financial_agent,
        message="Hello! I'm ready to help you analyze financial data and create CSV reports. What would you like to do?"
    )

if __name__ == "__main__":
    # Example usage
    print("Financial Analysis Tool - Example Usage")
    print("=" * 50)
    
    # Example 1: Get single ratio
    print("\n1. Getting P/E ratio for AAPL on a specific date:")
    result = get_financial_ratio_by_date("AAPL", "P/E Ratio", "2024-01-15")
    print(f"Result: {result.get('ratio_value', 'N/A')}")
    
    # Example 2: Create CSV (commented out to avoid file creation)
    print("\n2. Creating CSV example (function available):")
    print("create_financial_ratios_csv('AAPL', ['Close', 'P/E Ratio'], '2024-01-01', '2024-12-31')")
    
    print("\n3. Starting interactive chat...")
    run_financial_analysis()

    
    

#run_financial_analysis()
