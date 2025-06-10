import React, { useState, useEffect } from 'react';
import Layout from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Search, Download, Trash2, PlusCircle, Sparkles } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import axios from 'axios';

interface Stock {
  ticker: string;
  current_ratios: {
    'P/E Ratio': number;
    'P/B Ratio': number;
    'Current Ratio': number;
    'Debt-to-Equity': number;
    'Gross Margin': number;
    'Profit Margin': number;
    'Return on Equity (ROE)': number;
    'Return on Assets (ROA)': number;
    'EV/EBITDA': number;
  };
  time_series_ratios: {
    [date: string]: {
      'Stock Price': number;
    };
  };
}

interface ApiResponse {
  companies: {
    [ticker: string]: Stock;
  };
  price_trends: {
    [date: string]: {
      [ticker: string]: number;
    };
  };
}

interface FinancialDataPoint {
  metric: string;
  [ticker: string]: string | number;
}

interface RadarChartDataPoint {
  metric: string;
  fullMark: number;
  [ticker: string]: string | number;
}

const Compare = () => {
  const [selectedStocks, setSelectedStocks] = useState<Stock[]>([]);
  const [searchInput, setSearchInput] = useState<string>('');
  const [duration, setDuration] = useState<string>('1d');
  const [priceHistoryData, setPriceHistoryData] = useState<Array<{
    date: string;
    [ticker: string]: string | number;
  }>>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const stockColors: { [key: number]: string } = {
    0: '#ff6384',
    1: '#36a2eb',
    2: '#4bc0c0',
  };

  const durations: Array<{ value: string; label: string }> = [
    { value: '1d', label: '1 Day' },
    { value: '7d', label: '7 Days' },
    { value: '1month', label: '1 Month' },
    { value: '6month', label: '6 Months' },
    { value: '1yr', label: '1 Year' },
  ];

  const fetchStockData = async (tickers: string[], selectedDuration: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post<ApiResponse>('http://127.0.0.1:5000/detailed_comparison', {
        companies: tickers,
        duration: selectedDuration,
      });

      const stocks = Object.values(response.data.companies) as Stock[];
      setSelectedStocks(stocks);

      // Prepare price history data
      const priceData = Object.entries(response.data.price_trends).map(([date, prices]) => ({
        date,
        ...prices,
      }));
      setPriceHistoryData(priceData);
    } catch (err) {
      setError('Failed to fetch stock data. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedStocks.length > 0) {
      fetchStockData(selectedStocks.map(stock => stock.ticker), duration);
    }
  }, [duration]);

  const handleAddStock = (): void => {
    if (searchInput && selectedStocks.length < 3 && !selectedStocks.some(stock => stock.ticker.toLowerCase() === searchInput.toLowerCase())) {
      fetchStockData([...selectedStocks.map(s => s.ticker), searchInput.toUpperCase()], duration);
      setSearchInput('');
    }
  };

  const handleStockSelect = (ticker: string): void => {
    if (!selectedStocks.some(stock => stock.ticker === ticker)) {
      fetchStockData([...selectedStocks.map(stock => stock.ticker), ticker], duration);
    }
  };

  const handleRemoveStock = (ticker: string): void => {
    const newStocks = selectedStocks.filter(stock => stock.ticker !== ticker);
    setSelectedStocks(newStocks);
    if (newStocks.length > 0) {
      fetchStockData(newStocks.map(stock => stock.ticker), duration);
    } else {
      setPriceHistoryData([]);
    }
  };

  const handleDurationChange = (value: string): void => {
    setDuration(value);
    if (selectedStocks.length > 0) {
      fetchStockData(selectedStocks.map(s => s.ticker), value);
    }
  };

  // Financial data for bar chart
  const financialData: FinancialDataPoint[] = [
    {
      metric: 'P/E Ratio',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: stock.current_ratios['P/E Ratio'],
      }), {}),
    },
    {
      metric: 'Profit Margin',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: stock.current_ratios['Profit Margin'] * 100,
      }), {}),
    },
    {
      metric: 'ROE (%)',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: stock.current_ratios['Return on Equity (ROE)'] * 100,
      }), {}),
    },
  ];

  // Radar chart data
  const radarChartData: RadarChartDataPoint[] = [
    {
      metric: 'P/E',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: (stock.current_ratios['P/E Ratio'] / 50) * 100,
      }), {}),
      fullMark: 100,
    },
    {
      metric: 'ROE',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: (stock.current_ratios['Return on Equity (ROE)'] / 2) * 100,
      }), {}),
      fullMark: 100,
    },
    {
      metric: 'Profit Margin',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: (stock.current_ratios['Profit Margin'] / 0.5) * 100,
      }), {}),
      fullMark: 100,
    },
    {
      metric: 'Current Ratio',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: (stock.current_ratios['Current Ratio'] / 3) * 100,
      }), {}),
      fullMark: 100,
    },
    {
      metric: 'EV/EBITDA',
      ...selectedStocks.reduce<{ [key: string]: number }>((acc, stock) => ({
        ...acc,
        [stock.ticker]: (stock.current_ratios['EV/EBITDA'] / 30) * 100,
      }), {}),
      fullMark: 100,
    },
  ];

  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Stock Comparison</h1>
          <p className="text-muted-foreground">
            Compare multiple stocks side by side with AI-powered analysis
          </p>
        </div>

        {/* Stock Selection Area */}
        <Card>
          <CardHeader>
            <CardTitle>Select Stocks to Compare</CardTitle>
            <CardDescription>
              Choose up to 3 stocks for a detailed comparison
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  type="search"
                  placeholder="Enter stock ticker (e.g., AAPL)"
                  className="pl-9 w-full"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddStock()}
                />
              </div>
              <Button onClick={handleAddStock} className="gap-1">
                <PlusCircle className="h-4 w-4" />
                Add Stock
              </Button>
              <Select onValueChange={handleDurationChange} value={duration}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select duration" />
                </SelectTrigger>
                <SelectContent>
                  {durations.map(d => (
                    <SelectItem key={d.value} value={d.value}>{d.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {/* <Button variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </Button> */}
            </div>

            {error && <p className="text-red-500 mt-4">{error}</p>}

            <div className="flex flex-wrap gap-2 mt-4">
              {selectedStocks.map(stock => (
                <div key={stock.ticker} className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-primary/10 text-sm">
                  <span className="font-medium">{stock.ticker}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 ml-1 rounded-full"
                    onClick={() => handleRemoveStock(stock.ticker)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {loading && <p>Loading...</p>}

        {selectedStocks.length > 0 && !loading && (
          <>
            {/* Valuation Metrics Comparison */}
            <Card>
              <CardHeader>
                <CardTitle>Financial & Valuation Metrics</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[250px]">Metric</TableHead>
                        {selectedStocks.map(stock => (
                          <TableHead key={stock.ticker}>{stock.ticker}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {[
                        { label: 'P/E Ratio', key: 'P/E Ratio', format: (v: number) => v.toFixed(2) },
                        { label: 'P/B Ratio', key: 'P/B Ratio', format: (v: number) => v.toFixed(2) },
                        { label: 'Current Ratio', key: 'Current Ratio', format: (v: number) => v.toFixed(2) },
                        { label: 'Debt to Equity', key: 'Debt-to-Equity', format: (v: number) => v.toFixed(2) },
                        { label: 'Gross Margin (%)', key: 'Gross Margin', format: (v: number) => (v * 100).toFixed(2) + '%' },
                        { label: 'Profit Margin (%)', key: 'Profit Margin', format: (v: number) => (v * 100).toFixed(2) + '%' },
                        { label: 'ROE (%)', key: 'Return on Equity (ROE)', format: (v: number) => (v * 100).toFixed(2) + '%' },
                        { label: 'ROA (%)', key: 'Return on Assets (ROA)', format: (v: number) => (v * 100).toFixed(2) + '%' },
                        { label: 'EV/EBITDA', key: 'EV/EBITDA', format: (v: number) => v.toFixed(2) },
                      ].map(metric => (
                        <TableRow key={metric.label}>
                          <TableCell className="font-medium">{metric.label}</TableCell>
                          {selectedStocks.map(stock => (
                            <TableCell key={`${stock.ticker}-${metric.key}`}>
                              {metric.format(stock.current_ratios[metric.key as keyof typeof stock.current_ratios])}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>

            {/* Chart Visualizations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Stock Price Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Stock Price ({durations.find(d => d.value === duration)?.label})</CardTitle>
                  <CardDescription>Historical price performance comparison</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={priceHistoryData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <RechartsTooltip />
                        <Legend />
                        {selectedStocks.map((stock, idx) => (
                          <Line
                            key={stock.ticker}
                            type="monotone"
                            dataKey={stock.ticker}
                            name={stock.ticker}
                            stroke={stockColors[idx as keyof typeof stockColors]}
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            activeDot={{ r: 5 }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Financial Data Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Key Financial Metrics</CardTitle>
                  <CardDescription>Comparison of financial performance</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={financialData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis type="number" />
                        <YAxis dataKey="metric" type="category" width={120} />
                        <RechartsTooltip />
                        <Legend />
                        {selectedStocks.map((stock, idx) => (
                          <Bar
                            key={stock.ticker}
                            dataKey={stock.ticker}
                            name={stock.ticker}
                            fill={stockColors[idx as keyof typeof stockColors]}
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Radar Chart for Key Metrics */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Competitive Analysis</CardTitle>
                  <CardDescription>Relative comparison of key performance indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart outerRadius="70%" data={radarChartData}>
                        <PolarGrid stroke="#e2e8f0" />
                        <PolarAngleAxis dataKey="metric" />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} />
                        {selectedStocks.map((stock, idx) => (
                          <Radar
                            key={stock.ticker}
                            name={stock.ticker}
                            dataKey={stock.ticker}
                            stroke={stockColors[idx as keyof typeof stockColors]}
                            fill={stockColors[idx as keyof typeof stockColors]}
                            fillOpacity={0.3}
                          />
                        ))}
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </div>
    </Layout>
  );
};

export default Compare;