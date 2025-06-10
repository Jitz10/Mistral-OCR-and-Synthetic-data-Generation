import logging
import yfinance as yf
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import traceback
import asyncio
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import pandas as pd
import numpy as np
import google.generativeai as genai
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="alphasage.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DashboardAgent:
    def __init__(self):
        load_dotenv()
        self.logger = logger
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.mongo_client['alphasage']
        self.dashboard_collection = self.db['dashboards']
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')

    def _format_financial_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Format financial data into a more readable format."""
        if data is None or data.empty:
            return {}
        
        formatted_data = {}
        for column in data.columns:
            if isinstance(data[column].iloc[0], (int, float)):
                formatted_data[column] = data[column].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "N/A")
            else:
                formatted_data[column] = data[column].apply(lambda x: str(x) if pd.notnull(x) else "N/A")
        
        return formatted_data

    async def generate_dashboard_json(self, company_name: str, ticker: str) -> Dict[str, Any]:
        """Generate comprehensive dashboard JSON for a company."""
        try:
            # Check MongoDB for cached data
            cached_data = self._get_cached_dashboard(company_name, ticker)
            if cached_data:
                return cached_data

            # Get company data from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5y")
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            recommendations = stock.recommendations
            earnings = stock.earnings

            # Generate predictions using AI
            predictions = await self._generate_ai_predictions(
                company_name, ticker, info, financials, balance_sheet, cashflow, hist
            )

            # Generate investor thesis
            thesis = await self._generate_ai_investor_thesis(
                company_name, ticker, info, hist, recommendations, earnings, predictions
            )

            # Compile dashboard data
            dashboard_data = {
                "company_name": company_name,
                "about_company": {
                    "description": info.get('longBusinessSummary', 'N/A')
                },
                "about_sector": {
                    "description": f"The company operates in the {info.get('sector', 'N/A')} sector, specifically in {info.get('industry', 'N/A')}."
                },
                "key_ratios": self._extract_key_ratios(info),
                "price_and_volume_chart": {
                    "description": f"Latest price: ₹{hist['Close'].iloc[-1]:,.2f}, Volume: {hist['Volume'].iloc[-1]:,}"
                },
                "profit_and_loss_statement": self._format_financial_statement(financials),
                "balance_sheet": self._format_balance_sheet(balance_sheet),
                "cashflow_statement": self._format_cashflow_statement(cashflow),
                "predictions": predictions,
                "shareholders_information": self._get_shareholders_info(info),
                "competition": self._get_competition_info(info),
                "concall_presentation_summary": {
                    "description": self._get_concall_summary(recommendations)
                },
                "investor_thesis": thesis
            }

            # Store in MongoDB
            self._store_dashboard(company_name, ticker, dashboard_data)

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error generating dashboard: {str(e)}\n{traceback.format_exc()}")
            return {}

    def _extract_key_ratios(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format key financial ratios."""
        return {
            "enterprise_value": f"₹{info.get('enterpriseValue', 0):,.2f}",
            "revenue_growth_yoy": f"{info.get('revenueGrowth', 0)*100:.2f}%",
            "revenue_growth_3yr_cagr": f"{info.get('revenueGrowth', 0)*100:.2f}%",
            "pat_growth_yoy": f"{info.get('profitMargins', 0)*100:.2f}%",
            "pat_growth_3yr_cagr": f"{info.get('profitMargins', 0)*100:.2f}%",
            "ebitda_margin": f"{info.get('ebitdaMargins', 0)*100:.2f}%",
            "net_profit_margin": f"{info.get('profitMargins', 0)*100:.2f}%",
            "return_on_equity": f"{info.get('returnOnEquity', 0)*100:.2f}%",
            "return_on_capital_employed": f"{info.get('returnOnEquity', 0)*100:.2f}%",
            "return_on_invested_capital": f"{info.get('returnOnEquity', 0)*100:.2f}%",
            "accounts_receivable_days": info.get('daysSalesOutstanding', 0),
            "inventory_days": info.get('daysOfInventoryOnHand', 0),
            "accounts_payable_days": info.get('daysPayablesOutstanding', 0),
            "cash_conversion_cycle_days": info.get('cashConversionCycle', 0),
            "debt_to_equity_ratio": f"{info.get('debtToEquity', 0):.2f}x",
            "current_ratio": f"{info.get('currentRatio', 0):.2f}x",
            "quick_ratio": f"{info.get('quickRatio', 0):.2f}x",
            "interest_coverage_ratio": f"{info.get('interestCoverage', 0):.2f}x",
            "asset_turnover_ratio": f"{info.get('assetTurnover', 0):.2f}x",
            "degree_of_financial_leverage": f"{info.get('financialLeverage', 0):.2f}x"
        }

    def _format_financial_statement(self, financials: pd.DataFrame) -> Dict[str, Any]:
        """Format financial statement data."""
        if financials is None or financials.empty:
            return {"periods": []}

        periods = []
        for date in financials.columns:
            period_data = {
                "period": date.strftime("%b-%y"),
                "sales": f"₹{financials.loc['Total Revenue', date]:,.2f}" if 'Total Revenue' in financials.index else "N/A",
                "operating_profit": f"₹{financials.loc['Operating Income', date]:,.2f}" if 'Operating Income' in financials.index else "N/A",
                "net_profit": f"₹{financials.loc['Net Income', date]:,.2f}" if 'Net Income' in financials.index else "N/A"
            }
            periods.append(period_data)

        return {"periods": periods}

    def _format_balance_sheet(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Format balance sheet data."""
        if balance_sheet is None or balance_sheet.empty:
            return {"periods": []}

        periods = []
        for date in balance_sheet.columns:
            period_data = {
                "period": date.strftime("%b-%y"),
                "total_assets": f"₹{balance_sheet.loc['Total Assets', date]:,.2f}" if 'Total Assets' in balance_sheet.index else "N/A",
                "total_liabilities": f"₹{balance_sheet.loc['Total Liabilities', date]:,.2f}" if 'Total Liabilities' in balance_sheet.index else "N/A",
                "equity": f"₹{balance_sheet.loc['Total Stockholder Equity', date]:,.2f}" if 'Total Stockholder Equity' in balance_sheet.index else "N/A"
            }
            periods.append(period_data)

        return {"periods": periods}

    def _format_cashflow_statement(self, cashflow: pd.DataFrame) -> Dict[str, Any]:
        """Format cash flow statement data."""
        if cashflow is None or cashflow.empty:
            return {"periods": []}

        periods = []
        for date in cashflow.columns:
            period_data = {
                "period": date.strftime("%b-%y"),
                "cash_from_operating_activity": f"₹{cashflow.loc['Operating Cash Flow', date]:,.2f}" if 'Operating Cash Flow' in cashflow.index else "N/A",
                "cash_from_investing_activity": f"₹{cashflow.loc['Investing Cash Flow', date]:,.2f}" if 'Investing Cash Flow' in cashflow.index else "N/A",
                "cash_from_financing_activity": f"₹{cashflow.loc['Financing Cash Flow', date]:,.2f}" if 'Financing Cash Flow' in cashflow.index else "N/A"
            }
            periods.append(period_data)

        return {"periods": periods}

    def _get_shareholders_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Get shareholders information."""
        return {
            "shareholding_pattern": [
                {
                    "period": datetime.now().strftime("%b-%y"),
                    "promoter": f"{info.get('heldPercentInsiders', 0)*100:.2f}%",
                    "institutional": f"{info.get('heldPercentInstitutions', 0)*100:.2f}%",
                    "public": f"{(1 - info.get('heldPercentInsiders', 0) - info.get('heldPercentInstitutions', 0))*100:.2f}%"
                }
            ]
        }

    def _get_competition_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Get competition information."""
        return {
            "segments": [
                {
                    "segment": info.get('industry', 'N/A'),
                    "market_size": "N/A",
                    "beneficiaries": ["Competitor 1", "Competitor 2", "Competitor 3"]
                }
            ]
        }

    def _get_concall_summary(self, recommendations: pd.DataFrame) -> str:
        """Get concall summary from recommendations."""
        if recommendations is None or recommendations.empty:
            return "No recent analyst recommendations available."
        
        return f"Latest analyst recommendations: {recommendations.to_dict()}"

    def _get_cached_dashboard(self, company_name: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached dashboard data from MongoDB."""
        cached = self.dashboard_collection.find_one(
            {
                "company_name": company_name,
                "ticker": ticker,
                "created_at": {"$gte": datetime.now() - timedelta(days=2)}
            },
            sort=[("created_at", -1)]
        )
        return cached["dashboard_data"] if cached else None

    def _store_dashboard(self, company_name: str, ticker: str, dashboard_data: Dict[str, Any]) -> None:
        """Store dashboard data in MongoDB."""
        self.dashboard_collection.insert_one({
            "company_name": company_name,
            "ticker": ticker,
            "dashboard_data": dashboard_data,
            "created_at": datetime.now()
        })

    async def _generate_ai_predictions(self, company_name: str, ticker: str, info: Dict[str, Any],
                                     financials: Any, balance_sheet: Any, cashflow: Any,
                                     hist: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions using Gemini AI."""
        try:
            # Format historical data for AI analysis
            historical_data = {
                "financials": self._format_financial_data(financials),
                "balance_sheet": self._format_financial_data(balance_sheet),
                "cashflow": self._format_financial_data(cashflow),
                "price_history": hist.to_dict() if hist is not None else {},
                "company_info": info
            }

            prompt = f"""As a senior financial analyst, analyze {company_name} ({ticker}) and provide detailed predictions for 2026-2028.

Company Information:
- Market Cap: ₹{info.get('marketCap', 0):,.2f}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 0)*100:.2f}%
- Profit Margin: {info.get('profitMargins', 0)*100:.2f}%

Historical Data:
{json.dumps(historical_data, indent=2)}

Please provide predictions in the following JSON structure:
{{
    "predictions": {{
        "profit_and_loss_statement": {{
            "periods": [
                {{
                    "period": "Mar-2026",
                    "sales": "₹X,XXX Cr",
                    "yoy_growth": "XX%",
                    "expenses": "₹X,XXX Cr",
                    "operating_profit": "₹XXX Cr",
                    "opm": "XX.X%",
                    "other_income": "₹XX Cr",
                    "depreciation": "₹XX Cr",
                    "interest": "₹XX Cr",
                    "profit_before_tax": "₹XXX Cr",
                    "tax": "₹XX Cr",
                    "tax_rate": "XX%",
                    "net_profit": "₹XXX Cr",
                    "npm": "X.X%",
                    "yoy_growth_net_profit": "XX%"
                }},
                // Similar structure for 2027 and 2028
            ]
        }},
        "balance_sheet": {{
            "periods": [
                {{
                    "period": "Mar-2026",
                    "equity_share_capital": "₹X Cr",
                    "reserves": "₹XXX Cr",
                    "borrowings": "₹XXX Cr",
                    "yoy_growth_borrowings": "XX%",
                    "other_liabilities": "₹X,XXX Cr",
                    "total_liabilities": "₹X,XXX Cr",
                    "fixed_assets": "₹XXX Cr",
                    "capital_work_in_progress": "₹XX Cr",
                    "investments": "₹XX Cr",
                    "other_assets": "₹X,XXX Cr",
                    "total_assets": "₹X,XXX Cr",
                    "receivables": "₹XXX Cr",
                    "inventory": "₹XXX Cr",
                    "cash_and_bank": "₹XXX Cr",
                    "yoy_growth_cash": "XX%"
                }},
                // Similar structure for 2027 and 2028
            ]
        }},
        "cashflow_statement": {{
            "periods": [
                {{
                    "period": "Mar-2026",
                    "cash_from_operating_activity": "₹XXX Cr",
                    "cash_from_investing_activity": "₹XXX Cr",
                    "cash_from_financing_activity": "₹XXX Cr",
                    "net_cash_flow": "₹XXX Cr"
                }},
                // Similar structure for 2027 and 2028
            ]
        }}
    }},
    "key_ratios": {{
        "enterprise_value": "₹X,XXX Cr",
        "revenue_growth_yoy": "XX%",
        "revenue_growth_3yr_cagr": "XX%",
        "pat_growth_yoy": "XX%",
        "pat_growth_3yr_cagr": "XX%",
        "ebitda_margin": "XX.X%",
        "net_profit_margin": "X.X%",
        "return_on_equity": "XX.X%",
        "return_on_capital_employed": "XX.X%",
        "return_on_invested_capital": "XX.X%",
        "accounts_receivable_days": XX.X,
        "inventory_days": XX.X,
        "accounts_payable_days": XX.X,
        "cash_conversion_cycle_days": XX.X,
        "debt_to_equity_ratio": "X.XXx",
        "current_ratio": "X.XXx",
        "quick_ratio": "X.XXx",
        "interest_coverage_ratio": "X.XXx",
        "asset_turnover_ratio": "X.XXx",
        "degree_of_financial_leverage": "X.XXx"
    }},
    "competition": {{
        "segments": [
            {{
                "segment": "Segment Name",
                "voltage_range": "Voltage Range",
                "market_size": "₹X,XXX Cr",
                "beneficiaries": ["Competitor 1", "Competitor 2", "Competitor 3"]
            }}
        ],
        "market_by_industry": [
            {{
                "user_industry": "Industry Name",
                "voltage_class": "Voltage Class",
                "market_size": "₹XXX Bn",
                "competitors": ["Competitor 1", "Competitor 2", "Competitor 3"]
            }}
        ]
    }},
    "concall_presentation_summary": {{
        "description": "Detailed summary of recent concall and management commentary"
    }}
}}

Base your predictions on:
1. Historical performance trends
2. Industry growth rates
3. Market conditions
4. Company's strategic initiatives
5. Competitive position
6. Regulatory environment
7. Economic indicators

Provide specific numbers and percentages, and explain the reasoning behind your predictions."""

            # Get AI response
            response = self.model.generate_content(prompt)
            predictions = json.loads(response.text)
            
            return predictions

        except Exception as e:
            self.logger.error(f"Error generating AI predictions: {str(e)}")
            return {}

    async def _generate_ai_investor_thesis(self, company_name: str, ticker: str, info: Dict[str, Any],
                                         hist: pd.DataFrame, recommendations: Any, earnings: Any,
                                         predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive investor thesis using Gemini AI."""
        try:
            prompt = f"""As a senior investment analyst, provide a comprehensive investment thesis for {company_name} ({ticker}).

Company Information:
- Market Cap: ₹{info.get('marketCap', 0):,.2f}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 0)*100:.2f}%
- Profit Margin: {info.get('profitMargins', 0)*100:.2f}%

Historical Data:
- Price History: {hist.to_dict() if hist is not None else {}}
- Recommendations: {recommendations.to_dict() if recommendations is not None else {}}
- Earnings: {earnings.to_dict() if earnings is not None else {}}

Future Predictions:
{json.dumps(predictions, indent=2)}

Please provide your analysis in the following JSON structure:
{{
    "investment_summary": {{
        "current_price": "₹X,XXX",
        "target_price": "₹X,XXX",
        "upside_potential": "XX%",
        "risk_level": "Low/Medium/High",
        "investment_horizon": "Short/Medium/Long-term"
    }},
    "growth_metrics": {{
        "revenue_growth": "XX%",
        "profit_growth": "XX%",
        "historical_growth": {{
            "1_year": "XX%",
            "3_year": "XX%",
            "5_year": "XX%"
        }},
        "future_growth_projection": {{
            "projected_earnings_growth": "XX%",
            "projected_revenue_growth": "XX%",
            "projected_price_growth": "XX%"
        }}
    }},
    "market_position": {{
        "market_cap": "₹X,XXX Cr",
        "sector_rank": "X",
        "competitive_advantages": ["Advantage 1", "Advantage 2"],
        "market_share": "XX%"
    }},
    "financial_health": {{
        "debt_to_equity": "X.XX",
        "current_ratio": "X.XX",
        "profit_margins": "XX.X%",
        "return_on_equity": "XX.X%"
    }},
    "analyst_recommendations": {{
        "strong_buy": X,
        "buy": X,
        "hold": X,
        "sell": X,
        "strong_sell": X
    }},
    "risk_factors": [
        {{
            "category": "Market/Financial/Operational",
            "description": "Risk description",
            "severity": "Low/Medium/High"
        }}
    ],
    "investment_strategy": {{
        "entry_points": [
            {{
                "type": "Entry type",
                "price": "₹X,XXX",
                "confidence": "Low/Medium/High"
            }}
        ],
        "exit_strategy": {{
            "take_profit_levels": [
                {{"price": "₹X,XXX", "percentage": "XX%"}}
            ],
            "stop_loss_levels": [
                {{"price": "₹X,XXX", "percentage": "XX%"}}
            ],
            "trailing_stop": "XX%"
        }},
        "position_sizing": {{
            "suggested_position_size": "X-X% of portfolio",
            "risk_adjusted_allocation": "X% of portfolio",
            "maximum_position": "₹X,XXX"
        }}
    }}
}}

Provide specific numbers and percentages, and explain the reasoning behind your analysis."""

            # Get AI response
            response = self.model.generate_content(prompt)
            thesis = json.loads(response.text)
            
            return thesis

        except Exception as e:
            self.logger.error(f"Error generating AI investor thesis: {str(e)}")
            return {}

# Testing functionality
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Test data
    test_companies = [
        {
            "name": "Hitachi Energy India Ltd.",
            "ticker": "POWERINDIA.NS"
        },
        {
            "name": "Reliance Industries Ltd.",
            "ticker": "RELIANCE.NS"
        }
    ]
    
    async def run_tests():
        agent = DashboardAgent()
        for company in test_companies:
            try:
                print(f"\nTesting dashboard generation for {company['name']}")
                dashboard = await agent.generate_dashboard_json(company['name'], company['ticker'])
                
                # Save to results folder
                output_file = os.path.join("results", f"{company['ticker'].replace('.', '_')}_dashboard.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dashboard, f, indent=2)
                print(f"Saved dashboard to {output_file}")
                
            except Exception as e:
                print(f"Error testing {company['name']}: {str(e)}")
    
    # Run tests
    asyncio.run(run_tests()) 