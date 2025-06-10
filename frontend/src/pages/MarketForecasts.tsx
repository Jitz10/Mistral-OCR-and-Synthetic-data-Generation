import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Line, LineChart as RechartsLineChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from 'recharts';
import { AlertCircle, ArrowRight, BarChart, LineChart, Sparkles, TrendingUp } from 'lucide-react';

const forecastData = [
  { month: 'Jan', actual: 2400, forecast: 2400 },
  { month: 'Feb', actual: 2210, forecast: 2300 },
  { month: 'Mar', actual: 2290, forecast: 2200 },
  { month: 'Apr', actual: 2000, forecast: 2100 },
  { month: 'May', actual: 2181, forecast: 2050 },
  { month: 'Jun', actual: 2500, forecast: 2300 },
  { month: 'Jul', actual: 2100, forecast: 2200 },
  { month: 'Aug', actual: null, forecast: 2300 },
  { month: 'Sep', actual: null, forecast: 2400 },
  { month: 'Oct', actual: null, forecast: 2550 },
  { month: 'Nov', actual: null, forecast: 2650 },
  { month: 'Dec', actual: null, forecast: 2800 },
];

const MarketForecasts = () => {
  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-base">
              <LineChart className="h-4 w-4 mr-2" />
              S&P 500 Forecast
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart
                  data={forecastData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <RechartsTooltip />
                  <Line type="monotone" dataKey="actual" stroke="#8884d8" strokeWidth={2} />
                  <Line type="monotone" dataKey="forecast" stroke="#82ca9d" strokeWidth={2} strokeDasharray="5 5" />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 text-sm">
              <div className="flex items-start gap-1 mb-2">
                <Sparkles className="h-4 w-4 text-primary mt-0.5" />
                <p>
                  <strong>AI Analysis:</strong> Our model predicts a 12% upside for the S&P 500 by year-end, driven by cooling inflation and anticipated rate cuts.
                </p>
              </div>
              <div className="flex gap-3 text-xs text-muted-foreground mt-2">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-[#8884d8] mr-1" />
                  Historical
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-[#82ca9d] mr-1" />
                  Forecast
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-base">
              <BarChart className="h-4 w-4 mr-2" />
              Interest Rate Projection
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Current Rate</p>
                  <p className="text-2xl font-bold">5.50%</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-right">Year-End Projection</p>
                  <p className="text-2xl font-bold text-primary text-right">4.75%</p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>September 2025</span>
                  <span className="font-medium">5.25%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>November 2025</span>
                  <span className="font-medium">5.00%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>December 2025</span>
                  <span className="font-medium">4.75%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Q1 2026 (Est.)</span>
                  <span className="font-medium">4.50%</span>
                </div>
              </div>

              <div className="text-xs text-muted-foreground flex items-start gap-1">
                <AlertCircle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                <p>Model assigns 85% probability to at least 3 rate cuts by year-end based on latest economic data.</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-base">
              <TrendingUp className="h-4 w-4 mr-2" />
              Sector Performance Outlook
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-green-500 text-white mr-2">Bullish</Badge>
                  <span className="font-medium">Technology</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-green-500 text-white mr-2">Bullish</Badge>
                  <span className="font-medium">Healthcare</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-blue-500 text-white mr-2">Neutral</Badge>
                  <span className="font-medium">Financials</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-blue-500 text-white mr-2">Neutral</Badge>
                  <span className="font-medium">Consumer Staples</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-red-500 text-white mr-2">Bearish</Badge>
                  <span className="font-medium">Utilities</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Badge className="bg-red-500 text-white mr-2">Bearish</Badge>
                  <span className="font-medium">Energy</span>
                </div>
                <ArrowRight className="h-4 w-4" />
              </div>
            </div>

            <div className="mt-4 text-sm">
              <div className="flex items-start gap-1">
                <Sparkles className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                <p>
                  <strong>AI Insight:</strong> Technology sector expected to outperform amid AI-driven growth cycle. Consider underweighting utilities as rates remain elevated through Q3.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Economic Indicators Forecast</CardTitle>
          <CardDescription>AI model projections for key economic data points</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-4">Indicator</th>
                  <th className="text-left p-4">Current</th>
                  <th className="text-left p-4">Q3 2025</th>
                  <th className="text-left p-4">Q4 2025</th>
                  <th className="text-left p-4">Q1 2026</th>
                  <th className="text-left p-4">YoY Change</th>
                  <th className="text-left p-4">AI Confidence</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-4 font-medium">GDP Growth</td>
                  <td className="p-4">2.1%</td>
                  <td className="p-4">2.3%</td>
                  <td className="p-4">2.5%</td>
                  <td className="p-4">2.4%</td>
                  <td className="p-4 text-green-500">+0.4%</td>
                  <td className="p-4">
                    <Badge className="bg-green-100 text-green-800">High</Badge>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">CPI Inflation</td>
                  <td className="p-4">3.2%</td>
                  <td className="p-4">2.9%</td>
                  <td className="p-4">2.7%</td>
                  <td className="p-4">2.5%</td>
                  <td className="p-4 text-green-500">-0.7%</td>
                  <td className="p-4">
                    <Badge className="bg-green-100 text-green-800">High</Badge>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">Unemployment</td>
                  <td className="p-4">3.8%</td>
                  <td className="p-4">3.9%</td>
                  <td className="p-4">4.1%</td>
                  <td className="p-4">4.2%</td>
                  <td className="p-4 text-red-500">+0.4%</td>
                  <td className="p-4">
                    <Badge className="bg-amber-100 text-amber-800">Medium</Badge>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">10Y Treasury Yield</td>
                  <td className="p-4">4.6%</td>
                  <td className="p-4">4.3%</td>
                  <td className="p-4">4.1%</td>
                  <td className="p-4">3.9%</td>
                  <td className="p-4 text-green-500">-0.7%</td>
                  <td className="p-4">
                    <Badge className="bg-amber-100 text-amber-800">Medium</Badge>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-4 font-medium">Housing Starts</td>
                  <td className="p-4">1.39M</td>
                  <td className="p-4">1.42M</td>
                  <td className="p-4">1.45M</td>
                  <td className="p-4">1.47M</td>
                  <td className="p-4 text-green-500">+5.8%</td>
                  <td className="p-4">
                    <Badge className="bg-blue-100 text-blue-800">Low</Badge>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
        <CardFooter className="border-t py-4">
          <div className="flex items-start gap-1 text-sm">
            <Sparkles className="h-4 w-4 text-primary mt-0.5" />
            <p>
              <strong>AI Analysis:</strong> Economic indicators suggest a soft landing scenario with gradual disinflation and modest growth. Our models predict declining treasury yields in response to cooling inflation and moderating economic activity, creating a favorable environment for equities especially in sectors with high quality earnings and strong balance sheets.
            </p>
          </div>
        </CardFooter>
      </Card>
    </>
  );
};

export default MarketForecasts;