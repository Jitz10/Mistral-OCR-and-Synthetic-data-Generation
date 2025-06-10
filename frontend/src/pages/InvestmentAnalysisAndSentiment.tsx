import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Line, LineChart as RechartsLineChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from 'recharts';
import { AlertCircle, Briefcase, Building, Download, Share2, Sparkles, User } from 'lucide-react';

const sentimentData = [
  { date: 'Apr 6', score: 0.2 },
  { date: 'Apr 7', score: 0.3 },
  { date: 'Apr 8', score: -0.1 },
  { date: 'Apr 9', score: -0.4 },
  { date: 'Apr 10', score: -0.2 },
  { date: 'Apr 11', score: 0.1 },
  { date: 'Apr 12', score: 0.5 },
];

interface Props {
  tab: 'analysis' | 'sentiment';
}

const InvestmentAnalysisAndSentiment = ({ tab }: Props) => {
  return (
    <>
      {tab === 'analysis' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="col-span-1 md:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Investment Case Study: AI Semiconductor Market</CardTitle>
                <CardDescription>
                  Deep dive analysis on the AI chip industry and investment opportunities
                </CardDescription>
              </CardHeader>
              <CardContent className="prose prose-sm max-w-none">
                <h3>Industry Overview</h3>
                <p>
                  The AI semiconductor market has experienced unprecedented growth, with demand for specialized AI chips projected to grow at a 35% CAGR through 2030. Key players have seen their market capitalization expand significantly as data center operators and cloud service providers ramp up their AI infrastructure investments.
                </p>

                <h3>Market Leaders Analysis</h3>
                <p>
                  <strong>NVIDIA (NVDA)</strong> maintains dominant market share with its H100/H200 GPU architecture, controlling approximately 80% of the AI training chip market. Their software ecosystem (CUDA) creates significant switching costs and competitive moat. Current valuation at 42x forward earnings reflects strong growth expectations but leaves limited margin for execution errors.
                </p>

                <p>
                  <strong>AMD (AMD)</strong> has emerged as the primary challenger with its MI300 series, gaining traction in cloud deployments. Recent design wins at major hyperscalers suggest potential market share gains in 2025-2026. Trading at 32x forward earnings, valuation offers better risk/reward profile than market leader.
                </p>

                <h3>Supply Chain Considerations</h3>
                <p>
                  Backend testing and advanced packaging has emerged as a critical bottleneck in the AI chip supply chain. Companies specializing in high-bandwidth memory (HBM) integration and advanced packaging (<strong>ASE Technology, Amkor</strong>) may offer overlooked investment opportunities with more reasonable valuations (15-18x earnings).
                </p>

                <h3>Risk Assessment</h3>
                <ul>
                  <li><strong>Concentration risk:</strong> Hyperscaler spending patterns can create significant revenue volatility</li>
                  <li><strong>Competitive dynamics:</strong> Custom silicon development by major cloud providers poses long-term threat</li>
                  <li><strong>Valuation multiples:</strong> Current expectations leave minimal room for execution failures</li>
                  <li><strong>Regulatory/export controls:</strong> Geopolitical tensions could disrupt global supply chains</li>
                </ul>
              </CardContent>
              <CardFooter>
                <Button variant="outline" size="sm" className="mr-2">
                  <Download className="mr-2 h-4 w-4" />
                  Download Report
                </Button>
                <Button size="sm">
                  <Share2 className="mr-2 h-4 w-4" />
                  Share Analysis
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle>Earnings Calendar & AI Predictions</CardTitle>
                <CardDescription>
                  Upcoming earnings releases with AI model forecasts
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-3"></th>
                        <th className="text-left p-3">Date</th>
                        <th className="text-left p-3">Consensus EPS</th>
                        <th className="text-left p-3">AI Prediction</th>
                        <th className="text-left p-3">Deviation</th>
                        <th className="text-left p-3">Confidence</th>
                        <th className="text-left p-3">AI Commentary</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b">
                        <td className="p-3">
                          <div className="font-semibold">AMZN</div>
                          <div className="text-xs text-muted-foreground">Amazon.com</div>
                        </td>
                        <td className="p-3">Apr 25</td>
                        <td className="p-3">$0.83</td>
                        <td className="p-3 font-medium">$0.87</td>
                        <td className="p-3 text-green-500">+4.8%</td>
                        <td className="p-3">
                          <Badge className="bg-green-100 text-green-800">High</Badge>
                        </td>
                        <td className="p-3 text-sm">AWS growth acceleration likely</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-3">
                          <div className="font-semibold">MSFT</div>
                          <div className="text-xs text-muted-foreground">Microsoft</div>
                        </td>
                        <td className="p-3">Apr 30</td>
                        <td className="p-3">$2.82</td>
                        <td className="p-3 font-medium">$2.91</td>
                        <td className="p-3 text-green-500">+3.2%</td>
                        <td className="p-3">
                          <Badge className="bg-green-100 text-green-800">High</Badge>
                        </td>
                        <td className="p-3 text-sm">Copilot monetization better than expected</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-3">
                          <div className="font-semibold">GOOG</div>
                          <div className="text-xs text-muted-foreground">Alphabet</div>
                        </td>
                        <td className="p-3">Apr 30</td>
                        <td className="p-3">$1.89</td>
                        <td className="p-3 font-medium">$1.92</td>
                        <td className="p-3 text-green-500">+1.6%</td>
                        <td className="p-3">
                          <Badge className="bg-amber-100 text-amber-800">Medium</Badge>
                        </td>
                        <td className="p-3 text-sm">YouTube strength offsetting search pressure</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-3">
                          <div className="font-semibold">AAPL</div>
                          <div className="text-xs text-muted-foreground">Apple</div>
                        </td>
                        <td className="p-3">May 2</td>
                        <td className="p-3">$1.55</td>
                        <td className="p-3 font-medium">$1.52</td>
                        <td className="p-3 text-red-500">-1.9%</td>
                        <td className="p-3">
                          <Badge className="bg-amber-100 text-amber-800">Medium</Badge>
                        </td>
                        <td className="p-3 text-sm">China weakness may impact guidance</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-3">
                          <div className="font-semibold">META</div>
                          <div className="text-xs text-muted-foreground">Meta Platforms</div>
                        </td>
                        <td className="p-3">May 1</td>
                        <td className="p-3">$4.72</td>
                        <td className="p-3 font-medium">$4.93</td>
                        <td className="p-3 text-green-500">+4.4%</td>
                        <td className="p-3">
                          <Badge className="bg-green-100 text-green-800">High</Badge>
                        </td>
                        <td className="p-3 text-sm">Ad pricing improving, cost controls working</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="col-span-1 space-y-4">
            <Card className="bg-primary/[0.03]">
              <CardHeader className="pb-2">
                <CardTitle className="text flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  AI Weekly Insights
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="font-medium mb-1">Fed Policy Shift Likelihood: 78%</h3>
                  <p className="text-sm text-muted-foreground">Our NLP analysis of recent Fed communications suggests increasing dovish signals, with a rate cut likely by September FOMC meeting.</p>
                </div>

                <Separator />

                <div>
                  <h3 className="font-medium mb-1">Sector Rotation Detected</h3>
                  <p className="text-sm text-muted-foreground">Algorithmic analysis of fund flows indicates early rotation from growth to value sectors, particularly financials and industrials.</p>
                </div>

                <Separator />

                <div>
                  <h3 className="font-medium mb-1">Consumer Sentiment Divergence</h3>
                  <p className="text-sm text-muted-foreground">Social media sentiment analysis shows improving consumer outlook despite weaker retail spending, suggesting potential upside for consumer discretionary.</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Market Risk Monitor</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Volatility Index</span>
                      <span className="font-medium text-amber-500">Elevated</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-amber-500 w-[65%]"></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Credit Spreads</span>
                      <span className="font-medium text-green-500">Normal</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-green-500 w-[40%]"></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Liquidity Conditions</span>
                      <span className="font-medium text-green-500">Favorable</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-green-500 w-[35%]"></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Valuation Risk</span>
                      <span className="font-medium text-red-500">High</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-red-500 w-[80%]"></div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Geopolitical Risk</span>
                      <span className="font-medium text-amber-500">Moderate</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-amber-500 w-[55%]"></div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex items-start gap-1 text-sm">
                  <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                  <p>Overall risk assessment: <span className="font-medium">Moderate</span>. Focus on quality companies with strong balance sheets.</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Smart Money Movements</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="" />
                      <AvatarFallback className="bg-primary text-primary-foreground">13F</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="text-sm font-medium">Baupost Group</p>
                      <p className="text-xs text-muted-foreground">New position</p>
                    </div>
                  </div>
                  <Badge>AI Infrastructure</Badge>
                </div>

                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="" />
                      <AvatarFallback className="bg-primary text-primary-foreground">13F</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="text-sm font-medium">Renaissance Tech</p>
                      <p className="text-xs text-muted-foreground">Position increase</p>
                    </div>
                  </div>
                  <Badge>Semiconductors</Badge>
                </div>

                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="" />
                      <AvatarFallback className="bg-primary text-primary-foreground">13F</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="text-sm font-medium">Lone Pine Capital</p>
                      <p className="text-xs text-muted-foreground">Position reduction</p>
                    </div>
                  </div>
                  <Badge>E-commerce</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {tab === 'sentiment' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Market Sentiment Analysis</CardTitle>
                <CardDescription>
                  AI-powered analysis of news, social media, and analyst reports
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsLineChart
                      data={sentimentData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} />
                      <RechartsTooltip />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#8884d8"
                        strokeWidth={2}
                        dot={{ r: 6 }}
                        activeDot={{ r: 8 }}
                      />
                    </RechartsLineChart>
                  </ResponsiveContainer>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-sm font-medium">Social Media Sentiment</p>
                    <p className="text-2xl font-bold text-green-500">+0.42</p>
                    <p className="text-xs text-muted-foreground">Moderately Positive</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">News Sentiment</p>
                    <p className="text-2xl font-bold text-blue-500">+0.15</p>
                    <p className="text-xs text-muted-foreground">Neutral-Positive</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium">Analyst Sentiment</p>
                    <p className="text-2xl font-bold text-green-500">+0.63</p>
                    <p className="text-xs text-muted-foreground">Strongly Positive</p>
                  </div>
                </div>

                <div className="mt-6 text-sm">
                  <div className="flex items-start gap-1">
                    <Sparkles className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p>
                      <strong>AI Analysis:</strong> Market sentiment has shifted significantly positive over the past 48 hours following the Federal Reserve's latest commentary and better-than-expected unemployment data. Analyst notes are particularly optimistic on technology and financial sectors. Sentiment divergence between retail investors (more cautious) and institutional investors (more bullish) has widened to the largest gap in 3 months, historically a contrarian indicator.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-medium">Sentiment Leaders & Laggards</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium mb-2">Positive Sentiment Leaders</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-green-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <User className="h-4 w-4 text-green-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">NVDA</p>
                          <p className="text-xs text-muted-foreground">Nvidia Corp.</p>
                        </div>
                      </div>
                      <Badge className="bg-green-500 text-white">+0.87</Badge>
                    </div>

                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-green-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <Briefcase className="h-4 w-4 text-green-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">JPM</p>
                          <p className="text-xs text-muted-foreground">JPMorgan Chase</p>
                        </div>
                      </div>
                      <Badge className="bg-green-500 text-white">+0.82</Badge>
                    </div>

                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-green-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <Building className="h-4 w-4 text-green-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">MSFT</p>
                          <p className="text-xs text-muted-foreground">Microsoft Corp.</p>
                        </div>
                      </div>
                      <Badge className="bg-green-500 text-white">+0.78</Badge>
                    </div>
                  </div>
                </div>

                <Separator />

                <div>
                  <h3 className="text-sm font-medium mb-2">Negative Sentiment Laggards</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-red-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <User className="h-4 w-4 text-red-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">NFLX</p>
                          <p className="text-xs text-muted-foreground">Netflix, Inc.</p>
                        </div>
                      </div>
                      <Badge className="bg-red-500 text-white">-0.64</Badge>
                    </div>

                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-red-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <Building className="h-4 w-4 text-red-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">BA</p>
                          <p className="text-xs text-muted-foreground">Boeing Co.</p>
                        </div>
                      </div>
                      <Badge className="bg-red-500 text-white">-0.58</Badge>
                    </div>

                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="bg-red-500/10 h-8 w-8 rounded-full flex items-center justify-center">
                          <Briefcase className="h-4 w-4 text-red-500" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">XOM</p>
                          <p className="text-xs text-muted-foreground">Exxon Mobil</p>
                        </div>
                      </div>
                      <Badge className="bg-red-500 text-white">-0.51</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Keywords Trend Analysis</CardTitle>
                <CardDescription>
                  Frequently mentioned terms in financial news and social media
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">AI</span>
                    <span className="text-xs text-muted-foreground">+32%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">rate cut</span>
                    <span className="text-xs text-muted-foreground">+28%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">earnings beat</span>
                    <span className="text-xs text-muted-foreground">+18%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">semi chips</span>
                    <span className="text-xs text-muted-foreground">+15%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">cloud growth</span>
                    <span className="text-xs text-muted-foreground">+12%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">LLM training</span>
                    <span className="text-xs text-muted-foreground">+10%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">inflation data</span>
                    <span className="text-xs text-muted-foreground">+8%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">China demand</span>
                    <span className="text-xs text-muted-foreground">+5%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">retail sales</span>
                    <span className="text-xs text-muted-foreground">+3%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">oil supply</span>
                    <span className="text-xs text-muted-foreground">-8%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">recession risk</span>
                    <span className="text-xs text-muted-foreground">-12%</span>
                  </Badge>
                  <Badge className="text-lg py-1.5 px-3 bg-primary/10 hover:bg-primary/20 cursor-pointer">
                    <span className="mr-1">rate hike</span>
                    <span className="text-xs text-muted-foreground">-25%</span>
                  </Badge>
                </div>

                <div className="mt-6 text-sm">
                  <div className="flex items-start gap-1">
                    <Sparkles className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p>
                      <strong>AI Insight:</strong> Language patterns show significant shift toward optimism in monetary policy outlook, with "rate cut" mentions up 28% while "rate hike" mentions fell 25%. AI-related terms continue dominating financial discourse with 32% increase in mentions.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Contrarian Indicator Dashboard</CardTitle>
                <CardDescription>
                  Sentiment metrics that often signal market turning points
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Fear & Greed Index</span>
                      <span className="font-medium text-amber-500">Greed (72/100)</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-amber-500 w-[72%]"></div>
                    </div>
                    <p className="text-xs text-muted-foreground">Approaching extreme greed territory, historically a cautious signal</p>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>AAII Bull-Bear Spread</span>
                      <span className="font-medium text-red-500">+22% (Bearish Signal)</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-red-500 w-[80%]"></div>
                    </div>
                    <p className="text-xs text-muted-foreground">Retail investor bullishness at 6-month high, contrarian negative</p>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Put/Call Ratio</span>
                      <span className="font-medium text-green-500">0.72 (Bullish Signal)</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-green-500 w-[40%]"></div>
                    </div>
                    <p className="text-xs text-muted-foreground">Options positioning still showing healthy skepticism</p>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Fund Manager Cash Levels</span>
                      <span className="font-medium text-green-500">4.2% (Bullish Signal)</span>
                    </div>
                    <div className="h-2 bg-muted rounded overflow-hidden">
                      <div className="h-full bg-green-500 w-[35%]"></div>
                    </div>
                    <p className="text-xs text-muted-foreground">Institutional investors still holding above-average cash</p>
                  </div>
                </div>

                <div className="mt-2 text-sm">
                  <div className="flex items-start gap-1">
                    <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                    <p>
                      <strong>Contrarian Analysis:</strong> Mixed signals with retail optimism (bearish contrarian signal) offset by healthy institutional caution. Historically, this divergence has preceded short-term volatility but remained supportive of market uptrends over 3-6 month horizons.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </>
  );
};

export default InvestmentAnalysisAndSentiment;