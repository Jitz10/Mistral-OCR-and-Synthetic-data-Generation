import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Activity, AlertCircle, Download, RefreshCw, Sparkles } from 'lucide-react';

// AI recommendations
const rebalancingRecs = [
  { title: 'Technology Sector Overweight', description: 'Your portfolio has a 40% allocation to technology stocks, which is 12% above the benchmark. Consider trimming positions to reduce concentration risk.', severity: 'medium' },
  { title: 'Add Fixed Income Exposure', description: 'Portfolio lacks fixed income assets which could provide diversification benefits in volatile markets. Consider allocating 15-20% to bond ETFs.', severity: 'high' },
  { title: 'Increase International Exposure', description: 'Portfolio is heavily concentrated in US equities. Consider adding 10-15% allocation to international markets for geographical diversification.', severity: 'medium' },
  { title: 'Energy Sector Underweight', description: 'No exposure to Energy sector which has outperformed recently. Consider a 5% allocation to reduce correlation with technology sector.', severity: 'low' },
];

const AIInsightsSection = () => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'border-red-500/30 bg-red-500/5';
      case 'medium':
        return 'border-amber-500/30 bg-amber-500/5';
      case 'low':
        return 'border-blue-500/30 bg-blue-500/5';
      default:
        return '';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'medium':
        return <Activity className="h-4 w-4 text-amber-500" />;
      case 'low':
        return <RefreshCw className="h-4 w-4 text-blue-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* AI Rebalancing Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            AI Portfolio Recommendations
          </CardTitle>
          <CardDescription>
            Smart suggestions to optimize your investment strategy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {rebalancingRecs.map((rec, i) => (
            <div key={`rec-${i}`} className={`p-4 border rounded-lg ${getSeverityColor(rec.severity)}`}>
              <div className="flex items-center gap-2">
                {getSeverityIcon(rec.severity)}
                <h3 className="font-medium">{rec.title}</h3>
              </div>
              <p className="mt-2 text-sm">{rec.description}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Individual Stock Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Stock-Specific Insights
          </CardTitle>
          <CardDescription>
            AI analysis of your portfolio holdings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="p-4 border rounded-lg bg-primary/5">
            <div className="flex items-center gap-2">
              <div className="bg-primary/10 h-9 w-9 rounded-full flex items-center justify-center">
                <span className="font-semibold">AAPL</span>
              </div>
              <div>
                <h3 className="font-semibold">Apple Inc.</h3>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span className="text-green-500">+$805.50 (+20.7%)</span>
                  <span>•</span>
                  <span>25 shares</span>
                </div>
              </div>
              <Badge className="ml-auto bg-green-500 text-white">Buy</Badge>
            </div>
            <div className="mt-3 text-sm">
              <p>Apple continues to show strong momentum with Services revenue growing 18.5% YoY. AI integration in iOS 18 expected to drive the next upgrade cycle. Valuation at 31.5x forward earnings remains reasonable given growth profile and ecosystem strength. Technical indicators suggest strong support at $178 level.</p>
              <div className="mt-2 pt-2 border-t border-border/50 flex justify-between items-center">
                <div className="flex gap-2 text-xs text-muted-foreground">
                  <span className="font-medium">Conviction:</span> High
                  <span>•</span>
                  <span className="font-medium">Target:</span> $225 (+19.9%)
                </div>
                <Button variant="ghost" size="sm">View Analysis</Button>
              </div>
            </div>
          </div>

          <div className="p-4 border rounded-lg bg-amber-500/5">
            <div className="flex items-center gap-2">
              <div className="bg-amber-500/10 h-9 w-9 rounded-full flex items-center justify-center">
                <span className="font-semibold">TSLA</span>
              </div>
              <div>
                <h3 className="font-semibold">Tesla, Inc.</h3>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span className="text-red-500">-$584.65 (-12.5%)</span>
                  <span>•</span>
                  <span>15 shares</span>
                </div>
              </div>
              <Badge className="ml-auto bg-yellow-500 text-white">Hold</Badge>
            </div>
            <div className="mt-3 text-sm">
              <p>Tesla faces near-term headwinds with slowing EV demand and increased competition. Q2 deliveries expected to miss consensus estimates by 5-8%. However, energy storage business showing accelerating growth (+65% YoY) and FSD capabilities continue to improve. Maintain position but consider trimming on strength.</p>
              <div className="mt-2 pt-2 border-t border-border/50 flex justify-between items-center">
                <div className="flex gap-2 text-xs text-muted-foreground">
                  <span className="font-medium">Conviction:</span> Medium
                  <span>•</span>
                  <span className="font-medium">Support:</span> $245
                </div>
                <Button variant="ghost" size="sm">View Analysis</Button>
              </div>
            </div>
          </div>

          <Button variant="outline" className="w-full">
            View All Stock Insights
          </Button>
        </CardContent>
      </Card>

      {/* AI Market Commentary */}
      <Card className="bg-primary/[0.03]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            QuantAI Market Commentary
          </CardTitle>
        </CardHeader>
        <CardContent className="prose prose-sm max-w-none">
          <p><strong>Market Outlook (April 12, 2025):</strong> The S&P 500 continues to trade near all-time highs with notable sector divergence. Technology and Communication Services lead performance (+21.5% and +15.6% YTD respectively) while Energy remains the primary laggard (-4.2% YTD).</p>

          <p>Key drivers for your portfolio's outperformance include:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Technology overweight:</strong> Your 40% allocation (vs. benchmark 28%) has benefited from the sector's continued strength, particularly in AI-related names.</li>
            <li><strong>Quality bias:</strong> Holdings predominantly feature companies with strong balance sheets and robust free cash flow generation, characteristics that have outperformed in the current market environment.</li>
            <li><strong>Growth orientation:</strong> Growth factors have outpaced value metrics YTD, aligned with your portfolio's positioning.</li>
          </ul>

          <p>However, several risk factors are emerging that warrant attention:</p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Valuation concerns:</strong> Technology sector now trades at a 50% premium to historical averages, suggesting limited upside without earnings acceleration.</li>
            <li><strong>Concentration risk:</strong> Your top 3 holdings represent 48% of the portfolio versus 15% for the benchmark index.</li>
            <li><strong>Missing diversification:</strong> No exposure to Energy, Materials, or Utilities sectors limits downside protection in market rotations.</li>
          </ul>

          <p>Considering these factors, we recommend gradual rebalancing to reduce technology exposure while initiating positions in underrepresented sectors, particularly those with inflation-resistant characteristics and reasonable valuations.</p>
        </CardContent>
        <CardFooter>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Download Full Report
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default AIInsightsSection;