from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
import os
import tempfile
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from autogen_core import CancellationToken, MessageContext
from autogen_core.application import WorkerAgentId
from autogen_core.base import Agent, AgentId, AgentProxy
from autogen_core.components import RoutedAgent, TypeSubscription, message_handler
from autogen_core.tools import FunctionTool
import yfinance as yf
import pandas as pd
import traceback
from typing_extensions import Annotated

def get_config():
    """Retrieve the configuration for the AI model."""
    return [{
        "model": "llama-3.3-70b-versatile",
        "api_key": "*******",
        "api_type": "groq"
    }]

def create_code_writer(config_list):
    """Create a Code Writer Agent for CSV tasks."""
    system_message = """You are a helpful AI assistant.
    Your task is to manipulate a CSV file using Python and pandas.
    
    Tasks include:
    - Reading a CSV file (`tp.csv`)
    - Adding new rows or columns
    - Updating existing values
    - Writing the updated DataFrame back to `tp.csv`
    
    Always ensure to print the modified DataFrame.
    Use pandas best practices (e.g., `df.loc`, `df.append`, `df.to_csv`, etc.)
    
    Use only Python and pandas. Reply with TERMINATE when done.
    """

    return ConversableAgent(
        name="csv_code_writer_agent",
        system_message=system_message,
        llm_config={"config_list": config_list},
        code_execution_config=False,
    )

def create_code_executor():
    """Create a Code Executor Agent."""
    current_dir = os.getcwd()
    executor = LocalCommandLineCodeExecutor(
        timeout=10,
        work_dir=current_dir,
    )

    return ConversableAgent(
        name="csv_code_executor_agent",
        llm_config=False,
        code_execution_config={"executor": executor},
        human_input_mode="ALWAYS",
    )

def start_csv_conversation(writer_agent, executor_agent):
    task = """Read the file 'tp.csv' using pandas. 
    Then:
    - Add a column named 'status' with value 'active' for all rows.
    - Add a new row at the end with appropriate dummy values.
    - Save the file back to 'tp.csv' and print the updated DataFrame."""
    
    return executor_agent.initiate_chat(writer_agent, message=task)

if __name__ == "__main__":
    config_list = get_config()
    code_writer_agent = create_code_writer(config_list)
    code_executor_agent = create_code_executor()
    chat_result = start_csv_conversation(code_writer_agent, code_executor_agent)


class FinancialAnalysisRequest:
    """Request message for financial analysis"""
    def __init__(self, 
                 ticker: str, 
                 analysis_type: str = "comprehensive",
                 ratios: Optional[List[str]] = None,
                 duration: str = "1y",
                 comparison_tickers: Optional[List[str]] = None):
        self.ticker = ticker
        self.analysis_type = analysis_type  # "comprehensive", "valuation", "profitability", "custom"
        self.ratios = ratios or []
        self.duration = duration
        self.comparison_tickers = comparison_tickers or []

class FinancialAnalysisResponse:
    """Response message with analysis results"""
    def __init__(self, analysis_data: Dict[str, Any]):
        self.analysis_data = analysis_data

class FinancialAnalysisAgent(RoutedAgent):
    """
    Advanced Financial Analysis Agent that uses the financial ratio tool
    to provide comprehensive company analysis, ratio comparisons, and insights.
    """
    
    def __init__(self, description: str = "Financial Analysis Agent") -> None:
        super().__init__(description)
        self.available_ratios = {
            'valuation': ['P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'EV/Revenue', 'EV/EBITDA'],
            'profitability': ['Profit Margin', 'Gross Margin', 'Return on Assets (ROA)', 'Return on Equity (ROE)'],
            'leverage': ['Debt-to-Equity', 'Debt-to-Assets'],
            'liquidity': ['Current Ratio', 'Quick Ratio'],
            'dividend': ['Dividend Rate', 'Dividend Yield', 'Payout Ratio'],
            'market': ['Beta', 'Short Ratio'],
            'price': ['Open', 'High', 'Low', 'Close', 'Volume', 'Stock Price']
        }
        
        # Register the financial ratio tool
        self.financial_tool = stock_price_tool
    
    @message_handler
    async def handle_analysis_request(self, 
                                    message: FinancialAnalysisRequest, 
                                    ctx: MessageContext) -> FinancialAnalysisResponse:
        """Handle financial analysis requests"""
        try:
            analysis_result = await self._perform_analysis(
                ticker=message.ticker,
                analysis_type=message.analysis_type,
                ratios=message.ratios,
                duration=message.duration,
                comparison_tickers=message.comparison_tickers
            )
            
            return FinancialAnalysisResponse(analysis_result)
        
        except Exception as e:
            error_result = {
                "error": f"Analysis failed: {str(e)}",
                "ticker": message.ticker,
                "timestamp": datetime.now().isoformat()
            }
            return FinancialAnalysisResponse(error_result)
    
    async def _perform_analysis(self, 
                               ticker: str,
                               analysis_type: str,
                               ratios: List[str],
                               duration: str,
                               comparison_tickers: List[str]) -> Dict[str, Any]:
        """Perform the actual financial analysis"""
        
        analysis_result = {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "company_analysis": {},
            "comparison_analysis": {},
            "insights": [],
            "recommendations": []
        }
        
        # Get company basic info
        company_info = await self._get_company_info(ticker)
        analysis_result["company_info"] = company_info
        
        # Determine which ratios to analyze
        ratios_to_analyze = self._get_ratios_for_analysis(analysis_type, ratios)
        
        # Analyze primary company
        company_data = await self._analyze_company(ticker, ratios_to_analyze, duration)
        analysis_result["company_analysis"] = company_data
        
        # Analyze comparison companies if provided
        if comparison_tickers:
            comparison_data = {}
            for comp_ticker in comparison_tickers:
                comp_data = await self._analyze_company(comp_ticker, ratios_to_analyze, duration)
                comparison_data[comp_ticker] = comp_data
            analysis_result["comparison_analysis"] = comparison_data
        
        # Generate insights
        insights = self._generate_insights(analysis_result)
        analysis_result["insights"] = insights
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis_result)
        analysis_result["recommendations"] = recommendations
        
        return analysis_result
    
    async def _get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "employees": info.get("fullTimeEmployees", "N/A"),
                "website": info.get("website", "N/A"),
                "description": info.get("longBusinessSummary", "N/A")[:200] + "..." if info.get("longBusinessSummary") else "N/A"
            }
        except Exception as e:
            return {"error": f"Could not fetch company info: {str(e)}"}
    
    def _get_ratios_for_analysis(self, analysis_type: str, custom_ratios: List[str]) -> List[str]:
        """Determine which ratios to analyze based on analysis type"""
        if analysis_type == "custom" and custom_ratios:
            return custom_ratios
        elif analysis_type == "comprehensive":
            # Return a comprehensive set of key ratios
            return [
                'P/E Ratio', 'P/B Ratio', 'P/S Ratio',
                'Return on Equity (ROE)', 'Return on Assets (ROA)',
                'Profit Margin', 'Gross Margin',
                'Current Ratio', 'Debt-to-Equity',
                'Dividend Yield', 'Beta'
            ]
        elif analysis_type in self.available_ratios:
            return self.available_ratios[analysis_type]
        else:
            # Default to valuation ratios
            return self.available_ratios['valuation']
    
    async def _analyze_company(self, ticker: str, ratios: List[str], duration: str) -> Dict[str, Any]:
        """Analyze a single company for given ratios"""
        company_data = {
            "ticker": ticker,
            "ratios": {},
            "summary_stats": {},
            "trends": {}
        }
        
        for ratio in ratios:
            try:
                # Call the financial ratio tool
                ratio_data = get_financial_ratio(ticker, ratio, duration)
                
                if "error" not in ratio_data:
                    company_data["ratios"][ratio] = ratio_data
                    
                    # Calculate trend if we have time series data
                    if ratio_data.get("time_series"):
                        trend = self._calculate_trend(ratio_data["time_series"])
                        company_data["trends"][ratio] = trend
                else:
                    company_data["ratios"][ratio] = {"error": ratio_data["error"]}
                    
            except Exception as e:
                company_data["ratios"][ratio] = {"error": str(e)}
        
        return company_data
    
    def _calculate_trend(self, time_series: Dict[str, float]) -> Dict[str, Any]:
        """Calculate trend direction and percentage change"""
        if len(time_series) < 2:
            return {"trend": "insufficient_data"}
        
        dates = sorted(time_series.keys())
        first_value = time_series[dates[0]]
        last_value = time_series[dates[-1]]
        
        if first_value == 0:
            return {"trend": "undefined", "reason": "division_by_zero"}
        
        pct_change = ((last_value - first_value) / first_value) * 100
        
        trend_direction = "increasing" if pct_change > 5 else "decreasing" if pct_change < -5 else "stable"
        
        return {
            "trend": trend_direction,
            "percentage_change": round(pct_change, 2),
            "start_value": first_value,
            "end_value": last_value,
            "start_date": dates[0],
            "end_date": dates[-1]
        }
    
    def _generate_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate insights based on the analysis"""
        insights = []
        company_analysis = analysis_data.get("company_analysis", {})
        
        # Analyze trends
        trends = company_analysis.get("trends", {})
        for ratio, trend_data in trends.items():
            if trend_data.get("trend") == "increasing" and trend_data.get("percentage_change", 0) > 10:
                insights.append(f"{ratio} has increased significantly by {trend_data['percentage_change']:.1f}% over the period")
            elif trend_data.get("trend") == "decreasing" and trend_data.get("percentage_change", 0) < -10:
                insights.append(f"{ratio} has decreased significantly by {abs(trend_data['percentage_change']):.1f}% over the period")
        
        # Analyze ratio values
        ratios = company_analysis.get("ratios", {})
        
        # P/E Ratio insights
        if "P/E Ratio" in ratios and ratios["P/E Ratio"].get("current_value"):
            pe_ratio = ratios["P/E Ratio"]["current_value"]
            if pe_ratio > 30:
                insights.append(f"High P/E ratio of {pe_ratio:.1f} suggests the stock may be overvalued or investors expect high growth")
            elif pe_ratio < 10:
                insights.append(f"Low P/E ratio of {pe_ratio:.1f} suggests the stock may be undervalued or facing challenges")
        
        # ROE insights
        if "Return on Equity (ROE)" in ratios and ratios["Return on Equity (ROE)"].get("current_value"):
            roe = ratios["Return on Equity (ROE)"]["current_value"]
            if roe > 0.15:
                insights.append(f"Strong ROE of {roe*100:.1f}% indicates efficient use of shareholders' equity")
            elif roe < 0.05:
                insights.append(f"Low ROE of {roe*100:.1f}% may indicate poor profitability or inefficient equity use")
        
        # Debt insights
        if "Debt-to-Equity" in ratios and ratios["Debt-to-Equity"].get("current_value"):
            de_ratio = ratios["Debt-to-Equity"]["current_value"]
            if de_ratio > 2:
                insights.append(f"High debt-to-equity ratio of {de_ratio:.1f} indicates significant leverage and financial risk")
            elif de_ratio < 0.3:
                insights.append(f"Low debt-to-equity ratio of {de_ratio:.1f} suggests conservative financial management")
        
        return insights
    
    def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations based on analysis"""
        recommendations = []
        company_analysis = analysis_data.get("company_analysis", {})
        ratios = company_analysis.get("ratios", {})
        trends = company_analysis.get("trends", {})
        
        positive_signals = 0
        negative_signals = 0
        
        # Evaluate various factors
        if "Return on Equity (ROE)" in ratios:
            roe = ratios["Return on Equity (ROE)"].get("current_value", 0)
            if roe and roe > 0.15:
                positive_signals += 1
            elif roe and roe < 0.05:
                negative_signals += 1
        
        if "Current Ratio" in ratios:
            current_ratio = ratios["Current Ratio"].get("current_value", 0)
            if current_ratio and current_ratio > 1.5:
                positive_signals += 1
            elif current_ratio and current_ratio < 1:
                negative_signals += 1
        
        if "Profit Margin" in ratios:
            profit_margin = ratios["Profit Margin"].get("current_value", 0)
            if profit_margin and profit_margin > 0.1:
                positive_signals += 1
            elif profit_margin and profit_margin < 0.02:
                negative_signals += 1
        
        # Check trends
        positive_trends = sum(1 for trend in trends.values() 
                            if trend.get("trend") == "increasing" and trend.get("percentage_change", 0) > 5)
        negative_trends = sum(1 for trend in trends.values() 
                            if trend.get("trend") == "decreasing" and trend.get("percentage_change", 0) < -5)
        
        # Generate recommendation
        if positive_signals > negative_signals and positive_trends > negative_trends:
            recommendations.append("POSITIVE: Strong financial metrics and positive trends suggest this could be a good investment opportunity")
        elif negative_signals > positive_signals or negative_trends > positive_trends:
            recommendations.append("CAUTION: Weak financial metrics or negative trends suggest careful evaluation before investing")
        else:
            recommendations.append("NEUTRAL: Mixed signals - requires deeper analysis and consideration of market conditions")
        
        # Specific recommendations
        if positive_trends > 2:
            recommendations.append("Multiple improving financial ratios indicate positive momentum")
        
        if negative_trends > 2:
            recommendations.append("Multiple declining financial ratios warrant concern and further investigation")
        
        return recommendations

    async def analyze_company(self, 
                            ticker: str,
                            analysis_type: str = "comprehensive",
                            duration: str = "1y",
                            comparison_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Public method to analyze a company
        
        Args:
            ticker: Stock ticker symbol
            analysis_type: Type of analysis ("comprehensive", "valuation", "profitability", etc.)
            duration: Time period for analysis
            comparison_tickers: List of tickers to compare against
            
        Returns:
            Complete analysis results
        """
        request = FinancialAnalysisRequest(
            ticker=ticker,
            analysis_type=analysis_type,
            duration=duration,
            comparison_tickers=comparison_tickers
        )
        
        # Create a mock message context for direct calls
        class MockMessageContext:
            pass
        
        response = await self.handle_analysis_request(request, MockMessageContext())
        return response.analysis_data

    def get_available_ratios(self) -> Dict[str, List[str]]:
        """Return available ratio categories and ratios"""
        return self.available_ratios

    async def quick_valuation_check(self, ticker: str) -> Dict[str, Any]:
        """Quick valuation assessment using key ratios"""
        key_ratios = ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA']
        
        results = {}
        for ratio in key_ratios:
            try:
                data = get_financial_ratio(ticker, ratio, '1y')
                if "error" not in data:
                    results[ratio] = {
                        "current_value": data.get("current_value"),
                        "avg_value": data.get("statistics", {}).get("mean"),
                        "trend": self._calculate_trend(data.get("time_series", {}))
                    }
            except Exception as e:
                results[ratio] = {"error": str(e)}
        
        return {
            "ticker": ticker,
            "valuation_ratios": results,
            "timestamp": datetime.now().isoformat()
        }

# Example usage and testing functions
async def example_usage():
    """Example of how to use the Financial Analysis Agent"""
    
    # Create the agent
    agent = FinancialAnalysisAgent("Advanced Financial Analysis Agent")
    
    # Example 1: Comprehensive analysis of Apple
    print("=== Comprehensive Analysis of AAPL ===")
    apple_analysis = await agent.analyze_company("AAPL", "comprehensive", "1y")
    print(json.dumps(apple_analysis, indent=2, default=str))
    
    # Example 2: Valuation comparison between Apple and Microsoft
    print("\n=== Valuation Comparison: AAPL vs MSFT ===")
    comparison_analysis = await agent.analyze_company(
        "AAPL", 
        "valuation", 
        "1y", 
        ["MSFT"]
    )
    print(json.dumps(comparison_analysis, indent=2, default=str))
    
    # Example 3: Quick valuation check
    print("\n=== Quick Valuation Check for GOOGL ===")
    quick_check = await agent.quick_valuation_check("GOOGL")
    print(json.dumps(quick_check, indent=2, default=str))
    
    # Example 4: Custom ratio analysis
    print("\n=== Custom Analysis with Specific Ratios ===")
    custom_analysis = await agent.analyze_company(
        "TSLA",
        "custom",
        "6mo",
        ratios=['P/E Ratio', 'Beta', 'Profit Margin', 'Current Ratio']
    )
    print(json.dumps(custom_analysis, indent=2, default=str))

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())