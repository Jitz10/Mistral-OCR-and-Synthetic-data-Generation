// src/pages/PortfolioSection.tsx
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis, CartesianGrid, Legend, Area, AreaChart as RechartsAreaChart } from 'recharts';
import { ArrowUp, ArrowDown, Download, Sparkles, Trash2, Activity } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Utility functions
const getChangeColor = (change: number) => {
  return change >= 0 ? 'text-green-500' : 'text-red-500';
};

const getChangeIcon = (change: number) => {
  return change >= 0 ? <ArrowUp size={14} /> : <ArrowDown size={14} />;
};

const getRecommendationBadge = (rec: string) => {
  switch (rec) {
    case 'Strong Buy':
      return <Badge className="bg-emerald-500 text-white">Strong Buy</Badge>;
    case 'Buy':
      return <Badge className="bg-green-500 text-white">Buy</Badge>;
    case 'Hold':
      return <Badge className="bg-yellow-500 text-white">Hold</Badge>;
    case 'Sell':
      return <Badge className="bg-red-500 text-white">Sell</Badge>;
    default:
      return <Badge variant="outline">No Data</Badge>;
  }
};

interface Stock {
  ticker: string;
  name: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  pl: number;
  plPercent: number;
  aiRec: string;
  sector: string;
}

interface PortfolioSectionProps {
  portfolioHoldings: Stock[];
  setPortfolioHoldings: React.Dispatch<React.SetStateAction<Stock[]>>;
}

const PortfolioSection = ({ portfolioHoldings, setPortfolioHoldings }: PortfolioSectionProps) => {
  const { toast } = useToast();

  // Calculate total portfolio value
  const totalPortfolioValue = portfolioHoldings.reduce((sum, holding) => sum + holding.marketValue, 0);

  // Calculate allocation data dynamically
  const allocationData = portfolioHoldings
    .reduce((acc, holding) => {
      const existing = acc.find(item => item.name === holding.sector);
      if (existing) {
        existing.value += holding.marketValue;
      } else {
        acc.push({ name: holding.sector, value: holding.marketValue });
      }
      return acc;
    }, [] as { name: string; value: number }[])
    .map((item, index) => ({
      name: item.name,
      value: totalPortfolioValue ? (item.value / totalPortfolioValue) * 100 : 0,
      color: ['#3B82F6', '#EC4899', '#6366F1', '#10B981', '#14B8A6', '#F59E0B', '#EF4444', '#8B5CF6'][index % 8],
    }));

  const handleRemoveStock = (ticker: string) => {
    setPortfolioHoldings(prev => prev.filter(holding => holding.ticker !== ticker));
    toast({
      title: 'Stock Removed',
      description: `${ticker} has been removed from your portfolio.`,
    });
  };

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Portfolio Value</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${totalPortfolioValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Updated: Apr 12, 2025, 4:00 PM ET
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Day Change</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500 flex items-center">
              <ArrowUp size={20} className="mr-1" /> $186.42 (0.73%)
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              vs. S&P 500: +0.45%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Return</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500 flex items-center">
              <ArrowUp size={20} className="mr-1" /> $2,269.37 (9.4%)
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Since inception (Jan 15, 2025)
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Risk Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-between items-center">
              <div className="text-2xl font-bold">78/100</div>
              <div className="h-10 w-10 rounded-full bg-amber-500/10 flex items-center justify-center">
                <Activity className="h-5 w-5 text-amber-500" />
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Higher risk profile than benchmark
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Asset Allocation and Holdings */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Asset Allocation</CardTitle>
            <CardDescription>
              Portfolio breakdown by sector
            </CardDescription>
          </CardHeader>
          <CardContent>
            {portfolioHoldings.length > 0 ? (
              <>
                <div className="h-[250px] flex justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={allocationData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        nameKey="name"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                      >
                        {allocationData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <RechartsTooltip
                        formatter={(value, name) => [
                          `${(value as number).toFixed(1)}%`,
                          name,
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2 mt-2 text-xs">
                  {allocationData.map((item, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-3 h-3 mr-2 rounded-sm" style={{ backgroundColor: item.color }}></div>
                        <span>{item.name}</span>
                      </div>
                      <span className="font-medium">{item.value.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No holdings to display.</p>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Portfolio Holdings</CardTitle>
            <CardDescription>
              All stocks in your portfolio
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">Ticker</TableHead>
                    <TableHead>Shares</TableHead>
                    <TableHead className="text-right">Price</TableHead>
                    <TableHead className="text-right">Market Value</TableHead>
                    <TableHead className="text-right">P/L</TableHead>
                    <TableHead className="text-right">% P/L</TableHead>
                    <TableHead>AI Rating</TableHead>
                    <TableHead className="w-[80px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {portfolioHoldings.length > 0 ? (
                    portfolioHoldings.map(holding => (
                      <TableRow key={holding.ticker}>
                        <TableCell>
                          <div>
                            <div className="font-medium">{holding.ticker}</div>
                            <div className="text-xs text-muted-foreground">{holding.name}</div>
                          </div>
                        </TableCell>
                        <TableCell>{holding.shares}</TableCell>
                        <TableCell className="text-right">${holding.currentPrice.toFixed(2)}</TableCell>
                        <TableCell className="text-right">${holding.marketValue.toFixed(2)}</TableCell>
                        <TableCell className={`text-right ${getChangeColor(holding.pl)}`}>
                          ${Math.abs(holding.pl).toFixed(2)}
                        </TableCell>
                        <TableCell className={`text-right ${getChangeColor(holding.plPercent)}`}>
                          <div className="flex items-center justify-end gap-1">
                            {getChangeIcon(holding.plPercent)}
                            <span>{Math.abs(holding.plPercent).toFixed(1)}%</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          {getRecommendationBadge(holding.aiRec)}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleRemoveStock(holding.ticker)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={8} className="text-center">
                        No stocks in portfolio.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Performance</CardTitle>
          <CardDescription>
            Historical performance vs. benchmark
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsAreaChart
                data={[
                  { date: '1/15/2025', portfolio: 25000, benchmark: 25000 },
                  { date: '2/1/2025', portfolio: 25800, benchmark: 25400 },
                  { date: '2/15/2025', portfolio: 26100, benchmark: 25800 },
                  { date: '3/1/2025', portfolio: 25500, benchmark: 25300 },
                  { date: '3/15/2025', portfolio: 26300, benchmark: 25600 },
                  { date: '4/1/2025', portfolio: 26900, benchmark: 25900 },
                  { date: '4/12/2025', portfolio: 27200, benchmark: 26100 },
                ]}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EC4899" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#EC4899" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" />
                <YAxis />
                <CartesianGrid strokeDasharray="3 3" />
                <RechartsTooltip />
                <Legend />
                <Area type="monotone" dataKey="portfolio" stroke="#3B82F6" fillOpacity={1} fill="url(#colorPortfolio)" name="My Portfolio" />
                <Area type="monotone" dataKey="benchmark" stroke="#EC4899" fillOpacity={1} fill="url(#colorBenchmark)" name="S&P 500" />
              </RechartsAreaChart>
            </ResponsiveContainer>
          </div>
          <div className="text-xs text-muted-foreground mt-4 flex items-start gap-1">
            <Sparkles className="h-3.5 w-3.5 mt-0.5 text-primary flex-shrink-0" />
            <p>
              <strong>AI Analysis:</strong> Your portfolio has outperformed the S&P 500 by 4.2% since inception, driven primarily by strong performance in technology holdings (AAPL, MSFT) and consumer discretionary (AMZN). Portfolio volatility is 15% higher than the benchmark due to sector concentration. Key contributors to outperformance: AAPL (+20.7%), AMZN (+19.4%), and MSFT (+18.4%).
            </p>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          {/* <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export Data
          </Button> */}
          {/* <div className="flex gap-2">
            <Button variant="outline" size="sm">1W</Button>
            <Button variant="outline" size="sm">1M</Button>
            <Button variant="outline" size="sm">3M</Button>
            <Button variant="default" size="sm">YTD</Button>
            <Button variant="outline" size="sm">1Y</Button>
            <Button variant="outline" size="sm">ALL</Button>
          </div> */}
        </CardFooter>
      </Card>
    </div>
  );
};

export default PortfolioSection;