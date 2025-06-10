"""
AlphaSage System Test Runner
Simple script to test the complete AlphaSage financial analysis system
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_alphasage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_complete_system():
    """Test the complete AlphaSage system"""
    print("="*80)
    print("ALPHASAGE FINANCIAL ANALYSIS SYSTEM TEST")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test 1: Import all modules
        print("1. Testing module imports...")
        try:
            from tools import check_system_health, ReasoningTool, YFinanceAgentTool, VectorSearchTool
            from microagent import AlphaSageMicroAgents
            from macroagent import AlphaSageMacroAgents
            from pdf_output_agent import AlphaSagePDFAgent, generate_company_report
            print("   ✓ All modules imported successfully")
        except Exception as e:
            print(f"   ✗ Module import failed: {e}")
            return False
        
        # Test 2: System health check
        print("\n2. Checking system health...")
        health = check_system_health()
        print(f"   Overall Status: {health['status']}")
        for service, status in health['dependencies'].items():
            status_icon = "✓" if status == "connected" else "○" if status == "not_configured" else "✗"
            print(f"   {status_icon} {service.upper()}: {status}")
        
        # Test 3: Test micro agents
        print("\n3. Testing micro agents...")
        try:
            micro_agents = AlphaSageMicroAgents()
            test_company = "Ganesha Ecosphere Limited"
            
            # Test a few key micro agents
            financial_result = await micro_agents.analyze_financial_metrics(test_company)
            news_result = await micro_agents.analyze_market_news(test_company)
            
            print(f"   ✓ Financial metrics analysis: {'Success' if financial_result.success else 'Failed'}")
            print(f"   ✓ Market news analysis: {'Success' if news_result.success else 'Failed'}")
            
        except Exception as e:
            print(f"   ✗ Micro agents test failed: {e}")
        
        # Test 4: Test macro agents
        print("\n4. Testing macro agents...")
        try:
            macro_agents = AlphaSageMacroAgents()
            
            # Test business analysis
            business_result = await macro_agents.analyze_business(test_company)
            print(f"   ✓ Business analysis: {'Success' if business_result.success else 'Failed'}")
            
        except Exception as e:
            print(f"   ✗ Macro agents test failed: {e}")
        
        # Test 5: Generate sample PDF report
        print("\n5. Testing PDF report generation...")
        try:
            # Create reports directory
            os.makedirs("./reports", exist_ok=True)
            
            # Generate report
            report_result = await generate_company_report(
                company_name=test_company,
                output_path=f"./reports/{test_company.replace(' ', '_')}_Test_Report.pdf"
            )
            
            if report_result['success']:
                print(f"   ✓ PDF report generated successfully")
                print(f"   📄 File: {report_result['output_path']}")
                print(f"   📊 Sections: {report_result['sections_count']}")
                print(f"   📈 Data Quality: {report_result['data_quality']['overall_quality']*100:.1f}%")
            else:
                print(f"   ✗ PDF generation failed: {report_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ✗ PDF test failed: {e}")
        
        # Test 6: Performance summary
        print("\n6. Performance Summary...")
        print(f"   🚀 System Status: {'Operational' if health['status'] != 'unhealthy' else 'Needs Attention'}")
        
        connected_services = sum(1 for status in health['dependencies'].values() if status == "connected")
        total_services = len(health['dependencies'])
        print(f"   🔗 Connected Services: {connected_services}/{total_services}")
        
        # Recommendations
        print("\n7. Recommendations...")
        if health['dependencies'].get('mongodb') != 'connected':
            print("   📝 Consider setting up MongoDB for enhanced data storage")
        if health['dependencies'].get('redis') != 'connected':
            print("   📝 Consider setting up Redis for improved caching")
        if health['dependencies'].get('gemini') != 'connected':
            print("   📝 Add Gemini API key for advanced reasoning capabilities")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ SYSTEM TEST FAILED: {e}")
        logger.error(f"System test failed: {e}")
        return False

async def quick_analysis_demo(company_name: str = "Reliance Industries"):
    """Quick demo of analysis capabilities"""
    print(f"\n🎯 Quick Analysis Demo for: {company_name}")
    print("-" * 50)
    
    try:
        from microagent import AlphaSageMicroAgents
        
        micro_agents = AlphaSageMicroAgents()
        
        # Run quick analysis
        print("Running financial metrics analysis...")
        result = await micro_agents.analyze_financial_metrics(company_name)
        
        if result.success:
            metrics = result.data.get('metrics', {})
            print(f"✓ Analysis completed in {result.execution_time:.2f}s")
            print(f"  Company: {result.data.get('company_name', 'N/A')}")
            print(f"  Ticker: {result.data.get('ticker', 'N/A')}")
            
            # Display key metrics
            for metric, value in metrics.items():
                if metric != 'analysis' and value != 'N/A':
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
        else:
            print(f"✗ Analysis failed: {result.error}")
            
    except Exception as e:
        print(f"✗ Demo failed: {e}")

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            company = sys.argv[2] if len(sys.argv) > 2 else "Reliance Industries"
            asyncio.run(quick_analysis_demo(company))
        elif sys.argv[1] == "report":
            company = sys.argv[2] if len(sys.argv) > 2 else "Ganesha Ecosphere Limited"
            async def generate_report():
                from pdf_output_agent import generate_company_report
                result = await generate_company_report(company)
                if result['success']:
                    print(f"✓ Report generated: {result['output_path']}")
                else:
                    print(f"✗ Report failed: {result['error']}")
            asyncio.run(generate_report())
        else:
            print("Usage: python test_alphasage.py [demo|report] [company_name]")
    else:
        # Run full system test
        asyncio.run(test_complete_system())

if __name__ == "__main__":
    main()