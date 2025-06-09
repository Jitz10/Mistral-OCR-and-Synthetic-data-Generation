import os
import json
import logging
import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import all our modules
from macroagent import AlphaSageMacroAgents, MacroAgentResult
from microagent import AlphaSageMicroAgents, AgentResult
from tools import (
    ReasoningTool, YFinanceNumberTool, YFinanceNewsTool, 
    ArithmeticCalculationTool, VectorSearchRAGTool,
    check_system_health, cache_result, get_cached_result, generate_cache_key
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedTestResult:
    """Result structure for integrated testing"""
    test_name: str
    success: bool
    execution_time: float
    macro_agent_results: List[MacroAgentResult]
    micro_agent_results: List[AgentResult]
    tool_usage: Dict[str, int]
    data_flow: List[Dict[str, Any]]
    error: Optional[str] = None

class AlphaSageIntegratedTester:
    """
    Comprehensive integration tester for AlphaSage system
    Tests end-to-end functionality with macro agents, micro agents, and tools
    """
    
    def __init__(self):
        """Initialize the integrated tester"""
        self.macro_agents = AlphaSageMacroAgents()
        self.micro_agents = AlphaSageMicroAgents()
        self.system_health = check_system_health()
        
        # Test company details
        self.test_company = {
            "name": "Ganesha Ecosphere Limited",
            "ticker": "GANECOS.NS",
            "sector": "Environmental Services"
        }
        
        self.test_results = []
        self.tool_usage_stats = {}
        
        logger.info("AlphaSage Integrated Tester initialized")
        logger.info(f"System Health: {self.system_health}")
        logger.info(f"Test Company: {self.test_company['name']} ({self.test_company['ticker']})")

    def log_tool_usage(self, tool_name: str):
        """Track tool usage statistics"""
        if tool_name not in self.tool_usage_stats:
            self.tool_usage_stats[tool_name] = 0
        self.tool_usage_stats[tool_name] += 1

    async def test_business_analysis_integration(self) -> IntegratedTestResult:
        """Test complete business analysis using macro + micro agents + tools"""
        start_time = time.time()
        test_name = "Business Analysis Integration"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {test_name}")
        logger.info(f"{'='*60}")
        
        macro_results = []
        micro_results = []
        data_flow = []
        
        try:
            # Step 1: Use HistoricalDataAgent (Micro) to get company evolution
            logger.info("Step 1: Fetching historical data...")
            historical_result = await self.micro_agents.fetch_historical_data(
                ticker=self.test_company["ticker"],
                company_name=self.test_company["name"],
                years=5
            )
            micro_results.append(historical_result)
            self.log_tool_usage("HistoricalDataAgent")
            
            data_flow.append({
                "step": 1,
                "agent": "HistoricalDataAgent",
                "input": {"ticker": self.test_company["ticker"], "years": 5},
                "output_summary": f"Historical data points: {historical_result.data.get('data_points', 0)}",
                "success": historical_result.success
            })
            
            # Step 2: Use ValuationRatiosAgent (Micro) for financial metrics
            logger.info("Step 2: Calculating valuation ratios...")
            valuation_result = await self.micro_agents.calculate_valuation_ratios(
                ticker=self.test_company["ticker"],
                company_name=self.test_company["name"]
            )
            micro_results.append(valuation_result)
            self.log_tool_usage("ValuationRatiosAgent")
            
            data_flow.append({
                "step": 2,
                "agent": "ValuationRatiosAgent",
                "input": {"ticker": self.test_company["ticker"]},
                "output_summary": f"Ratios calculated: {len([k for k in valuation_result.data.keys() if k not in ['sources', 'timestamp', 'calculation_methods']])}",
                "success": valuation_result.success
            })
            
            # Step 3: Use BusinessResearchAgent (Macro) to synthesize analysis
            logger.info("Step 3: Conducting business research...")
            business_result = await self.macro_agents.analyze_business(
                company_name=self.test_company["name"]
            )
            macro_results.append(business_result)
            self.log_tool_usage("BusinessResearchAgent")
            
            data_flow.append({
                "step": 3,
                "agent": "BusinessResearchAgent",
                "input": {"company_name": self.test_company["name"]},
                "output_summary": f"Business model analysis completed, sources: {len(business_result.sources)}",
                "success": business_result.success
            })
            
            # Step 4: Use Tools directly for additional insights
            logger.info("Step 4: Getting recent news...")
            recent_news = await YFinanceNewsTool.fetch_company_news(
                ticker=self.test_company["ticker"],
                max_results=5
            )
            self.log_tool_usage("YFinanceNewsTool")
            
            data_flow.append({
                "step": 4,
                "agent": "YFinanceNewsTool",
                "input": {"ticker": self.test_company["ticker"]},
                "output_summary": f"News articles found: {len(recent_news)}",
                "success": len(recent_news) > 0
            })
            
            # Step 5: Use ReasoningTool to synthesize all data
            logger.info("Step 5: Synthesizing comprehensive analysis...")
            synthesis_data = {
                "company": self.test_company["name"],
                "historical_years": historical_result.data.get('years_analyzed', 0),
                "valuation_ratios": {k: v for k, v in valuation_result.data.items() if isinstance(v, (int, float))},
                "business_insights": business_result.data.get('business_model', 'No insights'),
                "news_count": len(recent_news)
            }
            
            synthesis_result = await ReasoningTool.reason_on_data(
                data=synthesis_data,
                query=f"Provide a comprehensive business analysis for {self.test_company['name']} based on historical performance, valuation ratios, business model, and recent news. What are the key investment highlights?"
            )
            self.log_tool_usage("ReasoningTool")
            
            data_flow.append({
                "step": 5,
                "agent": "ReasoningTool",
                "input": {"synthesis_data": "comprehensive_analysis"},
                "output_summary": f"Comprehensive analysis completed, confidence: {synthesis_result.get('confidence', 0)}",
                "success": bool(synthesis_result.get('reasoning'))
            })
            
            logger.info(f"✓ {test_name} completed successfully!")
            
            return IntegratedTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage={
                    "HistoricalDataAgent": 1,
                    "ValuationRatiosAgent": 1,
                    "BusinessResearchAgent": 1,
                    "YFinanceNewsTool": 1,
                    "ReasoningTool": 1
                },
                data_flow=data_flow
            )
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            return IntegratedTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow,
                error=str(e)
            )

    async def test_financial_health_analysis(self) -> IntegratedTestResult:
        """Test complete financial health analysis pipeline"""
        start_time = time.time()
        test_name = "Financial Health Analysis"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {test_name}")
        logger.info(f"{'='*60}")
        
        macro_results = []
        micro_results = []
        data_flow = []
        
        try:
            # Step 1: Calculate liquidity ratios
            logger.info("Step 1: Calculating liquidity ratios...")
            liquidity_result = await self.micro_agents.calculate_liquidity_ratios(
                company_name=self.test_company["name"]
            )
            micro_results.append(liquidity_result)
            
            data_flow.append({
                "step": 1,
                "agent": "LiquidityRatiosAgent",
                "output_summary": f"Current Ratio: {liquidity_result.data.get('Current_Ratio', 'N/A')}",
                "success": liquidity_result.success
            })
            
            # Step 2: Calculate leverage ratios
            logger.info("Step 2: Calculating leverage ratios...")
            leverage_result = await self.micro_agents.calculate_leverage_ratios(
                ticker=self.test_company["ticker"],
                company_name=self.test_company["name"]
            )
            micro_results.append(leverage_result)
            
            data_flow.append({
                "step": 2,
                "agent": "LeverageRatiosAgent",
                "output_summary": f"Debt-Equity: {leverage_result.data.get('Debt_Equity', 'N/A')}",
                "success": leverage_result.success
            })
            
            # Step 3: Calculate profitability ratios
            logger.info("Step 3: Calculating profitability ratios...")
            profitability_result = await self.micro_agents.calculate_profitability_ratios(
                ticker=self.test_company["ticker"],
                company_name=self.test_company["name"]
            )
            micro_results.append(profitability_result)
            
            data_flow.append({
                "step": 3,
                "agent": "ProfitabilityRatiosAgent",
                "output_summary": f"ROE: {profitability_result.data.get('ROE', 'N/A')}",
                "success": profitability_result.success
            })
            
            # Step 4: Macro agent debt and working capital analysis
            logger.info("Step 4: Macro debt and working capital analysis...")
            debt_wc_result = await self.macro_agents.analyze_debt_wc(
                company_name=self.test_company["name"]
            )
            macro_results.append(debt_wc_result)
            
            data_flow.append({
                "step": 4,
                "agent": "DebtAndWorkingCapitalAgent",
                "output_summary": f"Debt sustainability: {debt_wc_result.data.get('debt_sustainability', 'N/A')}",
                "success": debt_wc_result.success
            })
            
            # Step 5: Calculate financial health score using ArithmeticCalculationTool
            logger.info("Step 5: Calculating financial health score...")
            
            # Extract key metrics for scoring
            current_ratio = liquidity_result.data.get('Current_Ratio', 1.0)
            debt_equity = leverage_result.data.get('Debt_Equity', 1.0)
            roe = profitability_result.data.get('ROE', 0.1)
            
            # Convert string values to float if needed
            try:
                if isinstance(current_ratio, str):
                    current_ratio = float(current_ratio)
                if isinstance(debt_equity, str):
                    debt_equity = float(debt_equity)
                if isinstance(roe, str):
                    roe = float(roe)
            except:
                current_ratio, debt_equity, roe = 1.0, 1.0, 0.1
            
            health_calc = ArithmeticCalculationTool.calculate_metrics(
                inputs={
                    "Current_Ratio": current_ratio,
                    "Debt_Equity": debt_equity,
                    "ROE": roe
                },
                formula="Health_Score = (Current_Ratio * 0.3) + ((1 / (1 + Debt_Equity)) * 0.3) + (ROE * 0.4)"
            )
            
            health_score = 0.5  # Default
            if health_calc and health_calc.get('results'):
                health_score = health_calc['results'][0].get('value', 0.5)
            
            data_flow.append({
                "step": 5,
                "agent": "ArithmeticCalculationTool",
                "output_summary": f"Financial Health Score: {health_score:.3f}",
                "success": True
            })
            
            logger.info(f"✓ {test_name} completed successfully!")
            logger.info(f"Financial Health Score: {health_score:.3f}")
            
            return IntegratedTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage={
                    "LiquidityRatiosAgent": 1,
                    "LeverageRatiosAgent": 1,
                    "ProfitabilityRatiosAgent": 1,
                    "DebtAndWorkingCapitalAgent": 1,
                    "ArithmeticCalculationTool": 1
                },
                data_flow=data_flow
            )
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            return IntegratedTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow,
                error=str(e)
            )

    async def test_sector_and_risk_analysis(self) -> IntegratedTestResult:
        """Test sector analysis and risk assessment pipeline"""
        start_time = time.time()
        test_name = "Sector and Risk Analysis"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {test_name}")
        logger.info(f"{'='*60}")
        
        macro_results = []
        micro_results = []
        data_flow = []
        
        try:
            # Step 1: News analysis for sector trends
            logger.info("Step 1: Analyzing news for sector trends...")
            news_result = await self.micro_agents.analyze_news(
                ticker=self.test_company["ticker"]
            )
            micro_results.append(news_result)
            
            data_flow.append({
                "step": 1,
                "agent": "NewsAnalysisAgent",
                "output_summary": f"News articles analyzed: {news_result.data.get('news_count', 0)}",
                "success": news_result.success
            })
            
            # Step 2: Sentiment analysis
            logger.info("Step 2: Conducting sentiment analysis...")
            sentiment_result = await self.micro_agents.analyze_sentiment(
                company_name=self.test_company["name"],
                text=f"{self.test_company['sector']} sector analysis for {self.test_company['name']}"
            )
            micro_results.append(sentiment_result)
            
            data_flow.append({
                "step": 2,
                "agent": "SentimentAnalysisAgent",
                "output_summary": f"Sentiment: {sentiment_result.data.get('sentiment', 'neutral')}",
                "success": sentiment_result.success
            })
            
            # Step 3: Sector research using macro agent
            logger.info("Step 3: Conducting sector research...")
            sector_result = await self.macro_agents.analyze_sector(
                sector=self.test_company["sector"],
                company_name=self.test_company["name"]
            )
            macro_results.append(sector_result)
            
            data_flow.append({
                "step": 3,
                "agent": "SectorResearchAgent",
                "output_summary": f"Sector sentiment: {sector_result.data.get('sentiment_overview', 'neutral')}",
                "success": sector_result.success
            })
            
            # Step 4: Risk analysis using macro agent
            logger.info("Step 4: Conducting risk analysis...")
            risk_result = await self.macro_agents.analyze_risks(
                company_name=self.test_company["name"]
            )
            macro_results.append(risk_result)
            
            data_flow.append({
                "step": 4,
                "agent": "RiskAnalysisAgent",
                "output_summary": f"Risks identified: {len(risk_result.data.get('risks', []))}",
                "success": risk_result.success
            })
            
            # Step 5: Vector search for additional insights
            logger.info("Step 5: Searching for additional sector insights...")
            vector_results = await VectorSearchRAGTool.search_knowledge_base(
                query=f"{self.test_company['sector']} market trends regulatory environment",
                company_name=self.test_company["name"],
                max_results=3
            )
            
            data_flow.append({
                "step": 5,
                "agent": "VectorSearchRAGTool",
                "output_summary": f"Knowledge base results: {len(vector_results)}",
                "success": len(vector_results) > 0
            })
            
            logger.info(f"✓ {test_name} completed successfully!")
            
            return IntegratedTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage={
                    "NewsAnalysisAgent": 1,
                    "SentimentAnalysisAgent": 1,
                    "SectorResearchAgent": 1,
                    "RiskAnalysisAgent": 1,
                    "VectorSearchRAGTool": 1
                },
                data_flow=data_flow
            )
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            return IntegratedTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow,
                error=str(e)
            )

    async def test_future_predictions_pipeline(self) -> IntegratedTestResult:
        """Test future predictions and scenario analysis pipeline"""
        start_time = time.time()
        test_name = "Future Predictions Pipeline"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {test_name}")
        logger.info(f"{'='*60}")
        
        macro_results = []
        micro_results = []
        data_flow = []
        
        try:
            # Step 1: Extract guidance from micro agent
            logger.info("Step 1: Extracting management guidance...")
            guidance_result = await self.micro_agents.extract_guidance(
                company_name=self.test_company["name"]
            )
            micro_results.append(guidance_result)
            
            data_flow.append({
                "step": 1,
                "agent": "GuidanceExtractionAgent",
                "output_summary": f"Guidance confidence: {guidance_result.data.get('confidence', 0)}",
                "success": guidance_result.success
            })
            
            # Step 2: Generate scenarios using micro agent
            logger.info("Step 2: Generating scenarios...")
            base_metrics = {
                "revenue_growth": 12.0,
                "profit_margin": 8.5,
                "roe": 15.0,
                "debt_equity": 0.6
            }
            
            scenario_result = await self.micro_agents.generate_scenarios(
                company_name=self.test_company["name"],
                metrics=base_metrics
            )
            micro_results.append(scenario_result)
            
            data_flow.append({
                "step": 2,
                "agent": "ScenarioAnalysisAgent",
                "output_summary": f"Scenarios generated: bull, base, bear",
                "success": scenario_result.success
            })
            
            # Step 3: Future predictions using macro agent
            logger.info("Step 3: Generating future predictions...")
            predictions_result = await self.macro_agents.predict_future(
                company_name=self.test_company["name"],
                years=3
            )
            macro_results.append(predictions_result)
            
            data_flow.append({
                "step": 3,
                "agent": "FuturePredictionsAgent",
                "output_summary": f"Projections for {len(predictions_result.data.get('projections', []))} years",
                "success": predictions_result.success
            })
            
            # Step 4: Calculate projection accuracy score
            logger.info("Step 4: Calculating projection metrics...")
            
            projections = predictions_result.data.get('projections', [])
            confidence = predictions_result.data.get('confidence_level', 0.5)
            
            if projections:
                # Calculate average growth rate from projections
                revenues = [p.get('revenue', 0) for p in projections]
                if len(revenues) > 1:
                    growth_rates = []
                    for i in range(1, len(revenues)):
                        if revenues[i-1] > 0:
                            growth_rate = (revenues[i] - revenues[i-1]) / revenues[i-1]
                            growth_rates.append(growth_rate)
                    
                    if growth_rates:
                        avg_growth = sum(growth_rates) / len(growth_rates)
                        
                        accuracy_calc = ArithmeticCalculationTool.calculate_metrics(
                            inputs={
                                "Confidence": confidence,
                                "Avg_Growth": avg_growth,
                                "Scenarios_Available": 1 if scenario_result.success else 0
                            },
                            formula="Prediction_Score = (Confidence * 0.4) + (Avg_Growth * 0.3) + (Scenarios_Available * 0.3)"
                        )
                        
                        if accuracy_calc and accuracy_calc.get('results'):
                            prediction_score = accuracy_calc['results'][0].get('value', 0.5)
                        else:
                            prediction_score = confidence
                    else:
                        prediction_score = confidence
                else:
                    prediction_score = confidence
            else:
                prediction_score = 0.0
            
            data_flow.append({
                "step": 4,
                "agent": "ArithmeticCalculationTool",
                "output_summary": f"Prediction Score: {prediction_score:.3f}",
                "success": True
            })
            
            logger.info(f"✓ {test_name} completed successfully!")
            logger.info(f"Prediction Score: {prediction_score:.3f}")
            
            return IntegratedTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage={
                    "GuidanceExtractionAgent": 1,
                    "ScenarioAnalysisAgent": 1,
                    "FuturePredictionsAgent": 1,
                    "ArithmeticCalculationTool": 1
                },
                data_flow=data_flow
            )
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            return IntegratedTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow,
                error=str(e)
            )

    async def test_complete_analysis_pipeline(self) -> IntegratedTestResult:
        """Test the complete end-to-end analysis pipeline"""
        start_time = time.time()
        test_name = "Complete Analysis Pipeline"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting {test_name}")
        logger.info(f"{'='*80}")
        
        macro_results = []
        micro_results = []
        data_flow = []
        
        try:
            # Execute all macro agents
            macro_agents_to_test = [
                ("analyze_business", {"company_name": self.test_company["name"]}),
                ("analyze_sector", {"sector": self.test_company["sector"], "company_name": self.test_company["name"]}),
                ("deep_dive_company", {"company_name": self.test_company["name"]}),
                ("analyze_debt_wc", {"company_name": self.test_company["name"]}),
                ("analyze_current_affairs", {"company_name": self.test_company["name"]}),
                ("predict_future", {"company_name": self.test_company["name"], "years": 3}),
                ("analyze_concall", {"company_name": self.test_company["name"]}),
                ("analyze_risks", {"company_name": self.test_company["name"]})
            ]
            
            step = 1
            for agent_method, params in macro_agents_to_test:
                logger.info(f"Step {step}: Running {agent_method}...")
                
                method = getattr(self.macro_agents, agent_method)
                result = await method(**params)
                macro_results.append(result)
                
                data_flow.append({
                    "step": step,
                    "agent": agent_method,
                    "input": params,
                    "output_summary": f"Success: {result.success}, Sources: {len(result.sources)}, Tools: {len(result.tools_used)}",
                    "success": result.success,
                    "execution_time": result.execution_time
                })
                
                step += 1
            
            # Execute key micro agents
            micro_agents_to_test = [
                ("calculate_valuation_ratios", {"ticker": self.test_company["ticker"], "company_name": self.test_company["name"]}),
                ("calculate_profitability_ratios", {"ticker": self.test_company["ticker"], "company_name": self.test_company["name"]}),
                ("calculate_liquidity_ratios", {"company_name": self.test_company["name"]}),
                ("calculate_leverage_ratios", {"ticker": self.test_company["ticker"], "company_name": self.test_company["name"]}),
                ("analyze_news", {"ticker": self.test_company["ticker"]}),
                ("fetch_historical_data", {"ticker": self.test_company["ticker"], "company_name": self.test_company["name"], "years": 3})
            ]
            
            for agent_method, params in micro_agents_to_test:
                logger.info(f"Step {step}: Running {agent_method}...")
                
                method = getattr(self.micro_agents, agent_method)
                result = await method(**params)
                micro_results.append(result)
                
                data_flow.append({
                    "step": step,
                    "agent": agent_method,
                    "input": params,
                    "output_summary": f"Success: {result.success}, Sources: {len(result.sources)}",
                    "success": result.success,
                    "execution_time": result.execution_time
                })
                
                step += 1
            
            # Generate comprehensive summary using ReasoningTool
            logger.info(f"Step {step}: Generating comprehensive analysis...")
            
            # Aggregate all results
            summary_data = {
                "company": self.test_company["name"],
                "ticker": self.test_company["ticker"],
                "sector": self.test_company["sector"],
                "macro_agents_executed": len(macro_results),
                "micro_agents_executed": len(micro_results),
                "successful_macro": sum(1 for r in macro_results if r.success),
                "successful_micro": sum(1 for r in micro_results if r.success),
                "total_execution_time": sum(r.execution_time for r in macro_results + micro_results),
                "key_insights": []
            }
            
            # Extract key insights from successful results
            for result in macro_results:
                if result.success and result.data:
                    if 'business_model' in result.data:
                        summary_data["key_insights"].append(f"Business: {str(result.data['business_model'])[:100]}...")
                    elif 'trends' in result.data:
                        summary_data["key_insights"].append(f"Sector: {str(result.data['trends'])[:100]}...")
                    elif 'projections' in result.data:
                        projections = result.data['projections']
                        if projections:
                            summary_data["key_insights"].append(f"Future: {len(projections)} year projections generated")
            
            final_analysis = await ReasoningTool.reason_on_data(
                data=summary_data,
                query=f"Provide a comprehensive investment analysis summary for {self.test_company['name']} based on all the data gathered from macro and micro agents. What are the key investment highlights, risks, and recommendations?"
            )
            
            data_flow.append({
                "step": step,
                "agent": "ReasoningTool",
                "input": {"comprehensive_summary": True},
                "output_summary": f"Final analysis completed, confidence: {final_analysis.get('confidence', 0)}",
                "success": bool(final_analysis.get('reasoning'))
            })
            
            logger.info(f"✓ {test_name} completed successfully!")
            
            return IntegratedTestResult(
                test_name=test_name,
                success=True,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow
            )
            
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {str(e)}")
            return IntegratedTestResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                macro_agent_results=macro_results,
                micro_agent_results=micro_results,
                tool_usage=self.tool_usage_stats.copy(),
                data_flow=data_flow,
                error=str(e)
            )

    async def run_comprehensive_integration_tests(self):
        """Run all integration tests and generate comprehensive report"""
        print(f"\n{'='*100}")
        print("ALPHASAGE COMPREHENSIVE INTEGRATION TEST SUITE")
        print(f"{'='*100}")
        print(f"Test Company: {self.test_company['name']} ({self.test_company['ticker']})")
        print(f"Sector: {self.test_company['sector']}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System health check
        print(f"\n=== System Health Check ===")
        for service, status in self.system_health.items():
            status_text = "✓ ONLINE" if status else "✗ OFFLINE"
            print(f"{service.upper():15}: {status_text}")
        
        # Run all integration tests
        tests = [
            self.test_business_analysis_integration,
            self.test_financial_health_analysis,
            self.test_sector_and_risk_analysis,
            self.test_future_predictions_pipeline,
            self.test_complete_analysis_pipeline
        ]
        
        results = []
        for test_func in tests:
            try:
                result = await test_func()
                results.append(result)
                
                # Add small delay between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed with exception: {e}")
                results.append(IntegratedTestResult(
                    test_name=test_func.__name__,
                    success=False,
                    execution_time=0,
                    macro_agent_results=[],
                    micro_agent_results=[],
                    tool_usage={},
                    data_flow=[],
                    error=str(e)
                ))
        
        # Generate comprehensive report
        self.generate_integration_report(results)
        
        return results

    def generate_integration_report(self, results: List[IntegratedTestResult]):
        """Generate comprehensive integration test report"""
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE INTEGRATION TEST REPORT")
        print(f"{'='*100}")
        
        # Overall statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - successful_tests
        total_time = sum(r.execution_time for r in results)
        
        print(f"\n=== Overall Test Results ===")
        print(f"Total Integration Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests} (✓)")
        print(f"Failed Tests: {failed_tests} (✗)")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Average Test Time: {total_time/total_tests:.2f} seconds")
        
        # Test-by-test results
        print(f"\n=== Individual Test Results ===")
        for i, result in enumerate(results, 1):
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"{i}. {result.test_name:30} | {status:8} | {result.execution_time:6.2f}s")
            if not result.success:
                print(f"   Error: {result.error}")
        
        # Agent usage statistics
        all_macro_results = []
        all_micro_results = []
        for result in results:
            all_macro_results.extend(result.macro_agent_results)
            all_micro_results.extend(result.micro_agent_results)
        
        print(f"\n=== Agent Usage Statistics ===")
        print(f"Total Macro Agent Executions: {len(all_macro_results)}")
        print(f"Total Micro Agent Executions: {len(all_micro_results)}")
        
        # Macro agent success rates
        macro_stats = {}
        for result in all_macro_results:
            agent_name = result.agent_name
            if agent_name not in macro_stats:
                macro_stats[agent_name] = {'total': 0, 'success': 0, 'avg_time': 0}
            macro_stats[agent_name]['total'] += 1
            if result.success:
                macro_stats[agent_name]['success'] += 1
            macro_stats[agent_name]['avg_time'] += result.execution_time
        
        for agent, stats in macro_stats.items():
            stats['avg_time'] /= stats['total']
            success_rate = (stats['success'] / stats['total']) * 100
            print(f"  {agent:30}: {stats['success']}/{stats['total']} ({success_rate:5.1f}%) avg: {stats['avg_time']:.2f}s")
        
        # Micro agent success rates
        micro_stats = {}
        for result in all_micro_results:
            agent_name = result.agent_name
            if agent_name not in micro_stats:
                micro_stats[agent_name] = {'total': 0, 'success': 0, 'avg_time': 0}
            micro_stats[agent_name]['total'] += 1
            if result.success:
                micro_stats[agent_name]['success'] += 1
            micro_stats[agent_name]['avg_time'] += result.execution_time
        
        for agent, stats in micro_stats.items():
            stats['avg_time'] /= stats['total']
            success_rate = (stats['success'] / stats['total']) * 100
            print(f"  {agent:30}: {stats['success']}/{stats['total']} ({success_rate:5.1f}%) avg: {stats['avg_time']:.2f}s")
        
        # Tool usage summary
        print(f"\n=== Tool Usage Summary ===")
        for tool, count in sorted(self.tool_usage_stats.items()):
            print(f"  {tool:30}: {count} executions")
        
        # Data flow analysis
        print(f"\n=== Data Flow Analysis ===")
        total_steps = sum(len(r.data_flow) for r in results)
        successful_steps = sum(1 for r in results for step in r.data_flow if step['success'])
        
        print(f"Total Data Flow Steps: {total_steps}")
        print(f"Successful Steps: {successful_steps}")
        print(f"Data Flow Success Rate: {(successful_steps/total_steps)*100:.1f}%")
        
        # Recommendations
        print(f"\n=== Recommendations ===")
        if failed_tests > 0:
            print("• Review failed tests and address underlying issues")
        if not self.system_health['mongodb']:
            print("• Set up MongoDB for enhanced document search capabilities")
        if not self.system_health['redis']:
            print("• Set up Redis for improved caching and performance")
        if not self.system_health['gemini_api']:
            print("• Configure Gemini API key for advanced reasoning capabilities")
        if total_time > 120:  # 2 minutes
            print("• Consider optimizing slow agents for better performance")
        
        print(f"\n=== Summary for {self.test_company['name']} ===")
        print(f"✓ Business analysis completed using {len(macro_stats)} macro agents")
        print(f"✓ Financial metrics calculated using {len(micro_stats)} micro agents")
        print(f"✓ Data orchestration across {total_steps} processing steps")
        print(f"✓ Comprehensive integration testing completed in {total_time:.2f} seconds")
        
        print(f"\n{'='*100}")
        print("INTEGRATION TESTING COMPLETED")
        print(f"{'='*100}")

async def main():
    """Main function for comprehensive integration testing"""
    print("AlphaSage Integrated System Testing")
    print("Date: June 9, 2025")
    print("=" * 50)
    
    # Initialize integrated tester
    tester = AlphaSageIntegratedTester()
    
    # Run comprehensive integration tests
    results = await tester.run_comprehensive_integration_tests()
    
    # Save detailed results to file
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integration_test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {
                "test_name": result.test_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error,
                "macro_agent_count": len(result.macro_agent_results),
                "micro_agent_count": len(result.micro_agent_results),
                "tool_usage": result.tool_usage,
                "data_flow_steps": len(result.data_flow),
                "data_flow": result.data_flow
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump({
                "test_company": tester.test_company,
                "system_health": tester.system_health,
                "timestamp": timestamp,
                "total_tests": len(results),
                "successful_tests": sum(1 for r in results if r.success),
                "results": serializable_results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
    except Exception as e:
        print(f"\nWarning: Could not save results to file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
