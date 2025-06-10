import React, { useState } from 'react';
import Layout from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Activity, ArrowDown, ArrowDownLeft, ArrowRightCircle, ArrowUp, Compass, Award, Building, Factory, Globe, HardDrive, Monitor, PanelTop, ShoppingCart, Sparkles, Gauge, PiggyBank
} from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useQuery } from '@tanstack/react-query';
import { sectorSwotData } from '@/data/sectorSwotData';
import { industryDependenciesData } from '@/data/industryDependenciesData';
import { getESGRankings, getAISectorOutlook, generateDynamicOutlook } from '@/data/sectorData';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Area, AreaChart as RechartsAreaChart, ResponsiveContainer, Treemap } from 'recharts';

interface SectorData {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  color: string;
  fiftyDayAverage: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  fiftyTwoWeekChangePercent: number;
}

interface TopStockData {
  symbol: string;
  name: string;
  stock_price: number;
  percent_change: number;
  market_weight: number;
  pe_ratio: number;
  rating: string;
}

interface HeatmapData {
  name: string;
  size: number;
  value: number;
  color: string;
}

const Sectors = () => {
  const [selectedSector, setSelectedSector] = useState('technology');

  // Fetch sector info from API
  const { data: sectorInfo, isLoading: isSectorInfoLoading } = useQuery({
    queryKey: ['sectorInfo', selectedSector],
    queryFn: async () => {
      const response = await fetch('http://127.0.0.1:5000/sector_info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sector: selectedSector,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch sector info');
      }

      return response.json();
    },
    enabled: !!selectedSector,
  });

  // Fetch top companies for the sector
  const { data: sectorData, isLoading: isSectorLoading } = useQuery({
    queryKey: ['sector', selectedSector],
    queryFn: async () => {
      const response = await fetch('http://127.0.0.1:5000/top_companies_sector', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sector: selectedSector,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch sector data');
      }

      return response.json();
    },
    enabled: !!selectedSector,
  });

  const sectors: Record<string, SectorData> = {
    'basic-materials': {
      id: 'basic-materials',
      name: 'Basic Materials',
      icon: Factory,
      color: '#A3E635',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'communication-services': {
      id: 'communication-services',
      name: 'Communication Services',
      icon: Globe,
      color: '#14B8A6',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'consumer-cyclical': {
      id: 'consumer-cyclical',
      name: 'Consumer Cyclical',
      icon: ShoppingCart,
      color: '#EC4899',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'consumer-defensive': {
      id: 'consumer-defensive',
      name: 'Consumer Defensive',
      icon: ShoppingCart,
      color: '#8B5CF6',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'energy': {
      id: 'energy',
      name: 'Energy',
      icon: Gauge,
      color: '#F59E0B',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'financial-services': {
      id: 'financial-services',
      name: 'Financial Services',
      icon: PiggyBank,
      color: '#6366F1',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'healthcare': {
      id: 'healthcare',
      name: 'Healthcare',
      icon: Activity,
      color: '#10B981',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'industrials': {
      id: 'industrials',
      name: 'Industrials',
      icon: HardDrive,
      color: '#F97316',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'real-estate': {
      id: 'real-estate',
      name: 'Real Estate',
      icon: Building,
      color: '#0EA5E9',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'technology': {
      id: 'technology',
      name: 'Technology',
      icon: Monitor,
      color: '#3B82F6',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
    'utilities': {
      id: 'utilities',
      name: 'Utilities',
      icon: Gauge,
      color: '#8B5CF6',
      fiftyDayAverage: sectorInfo?.fiftyDayAverage || 0,
      fiftyTwoWeekHigh: sectorInfo?.fiftyTwoWeekHigh || 0,
      fiftyTwoWeekLow: sectorInfo?.fiftyTwoWeekLow || 0,
      fiftyTwoWeekChangePercent: sectorInfo?.fiftyTwoWeekChangePercent || 0,
    },
  };

  // Update topStocks to use API data
  const topStocks: Record<string, TopStockData[]> = {
    'basic-materials': sectorData || [],
    'communication-services': sectorData || [],
    'consumer-cyclical': sectorData || [],
    'consumer-defensive': sectorData || [],
    'energy': sectorData || [],
    'financial-services': sectorData || [],
    'healthcare': sectorData || [],
    'industrials': sectorData || [],
    'real-estate': sectorData || [],
    'technology': sectorData || [],
    'utilities': sectorData || [],
  };

  // Mock data for other sectors
  Object.keys(sectors).forEach(sector => {
    if (!topStocks[sector]) {
      topStocks[sector] = Array(10).fill(0).map((_, i) => ({
        symbol: `TICK${i}`,
        name: `Company ${i}`,
        stock_price: Math.random() * 500 + 50,
        percent_change: (Math.random() * 6 - 3),
        market_weight: Math.round(Math.random() * 500 + 50),
        pe_ratio: Math.random() * 40 + 10,
        rating: Math.random() > 0.5 ? 'Strong Buy' : 'Strong Sell',
      }));
    }
  });

  const renderSparkline = (data: number[]) => {
    return (
      <ChartContainer config={{}} className="h-8 w-16">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsAreaChart data={data.map((value, i) => ({ value, i }))} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="sparklineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3B82F6"
              fillOpacity={1}
              fill="url(#sparklineGradient)"
              isAnimationActive={false}
              strokeWidth={1.5}
            />
          </RechartsAreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'text-green-500' : 'text-red-500';
  };

  const getChangeIcon = (change: number) => {
    return change >= 0 ? <ArrowUp size={14} /> : <ArrowDown size={14} />;
  };

  // Generate heatmap data based on sector performance
  const getHeatmapData = (): HeatmapData[] => {
    return Object.values(sectors).map(sector => ({
      name: sector.name,
      size: sector.fiftyDayAverage || 1000, // Use fiftyDayAverage as a proxy for size
      value: sector.fiftyTwoWeekChangePercent,
      color: sector.fiftyTwoWeekChangePercent > 15 ? '#10B981' :
             sector.fiftyTwoWeekChangePercent > 10 ? '#22C55E' :
             sector.fiftyTwoWeekChangePercent > 5 ? '#A3E635' :
             sector.fiftyTwoWeekChangePercent > 0 ? '#FCD34D' :
             sector.fiftyTwoWeekChangePercent > -5 ? '#F97316' : '#EF4444',
    }));
  };

  const swotAnalysis = sectorSwotData;
  const industryDependencies = industryDependenciesData;
  const esgRankings = getESGRankings(selectedSector);
  const aiOutlook = getAISectorOutlook(selectedSector) ||
    generateDynamicOutlook(selectedSector, {
      fiftyTwoWeekChangePercent: sectors[selectedSector].fiftyTwoWeekChangePercent,
    });

  const currentSector = sectors[selectedSector];

  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Sector Analysis</h1>
          <p className="text-muted-foreground">
            Analyze performance, trends, and opportunities across market sectors
          </p>
        </div>

        {/* Sector Selection Tabs */}
        <Tabs defaultValue={selectedSector} onValueChange={setSelectedSector} className="w-full">
          <div className="flex overflow-x-auto no-scrollbar pb-2">
            <TabsList className="bg-background h-auto p-1 flex">
              {Object.values(sectors).map(sector => (
                <TabsTrigger key={sector.id} value={sector.id} className="flex items-center gap-2 py-2 px-4">
                  <sector.icon className="h-4 w-4" />
                  <span className="truncate">{sector.name}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          {Object.keys(sectors).map(sectorKey => (
            <TabsContent key={sectorKey} value={sectorKey} className="mt-4 space-y-6">
              {/* Sector Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">50-Day Average</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center">
                      <div className="text-2xl font-bold">{(sectors[sectorKey].fiftyDayAverage / 1000000).toFixed(2)}M</div>
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <Gauge className="h-5 w-5 text-primary" />
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Average index value over past 50 days
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">52-Week High</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center">
                      <div className="text-2xl font-bold">{(sectors[sectorKey].fiftyTwoWeekHigh / 1000000).toFixed(2)}M</div>
                      <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                        <ArrowUp className="h-5 w-5 text-green-500" />
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Highest index value in the past year
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">52-Week Low</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center">
                      <div className="text-2xl font-bold">{(sectors[sectorKey].fiftyTwoWeekLow / 1000000).toFixed(2)}M</div>
                      <div className="h-10 w-10 rounded-full bg-red-500/10 flex items-center justify-center">
                        <ArrowDown className="h-5 w-5 text-red-500" />
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Lowest index value in the past year
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">52-Week Change</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center">
                      <div className={`text-2xl font-bold ${getChangeColor(sectors[sectorKey].fiftyTwoWeekChangePercent)}`}>
                        {sectors[sectorKey].fiftyTwoWeekChangePercent >= 0 ? '+' : ''}{sectors[sectorKey].fiftyTwoWeekChangePercent.toFixed(2)}%
                      </div>
                      <div className={`h-10 w-10 rounded-full ${sectors[sectorKey].fiftyTwoWeekChangePercent >= 0 ? 'bg-green-500/10' : 'bg-red-500/10'} flex items-center justify-center`}>
                        {sectors[sectorKey].fiftyTwoWeekChangePercent >= 0 ?
                          <ArrowUp className="h-5 w-5 text-green-500" /> :
                          <ArrowDown className="h-5 w-5 text-red-500" />
                        }
                      </div>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      Percentage change over the past year
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Top Stocks Table */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between">
                    <span>Top Stocks in {sectors[sectorKey].name}</span>
                  </CardTitle>
                  <CardDescription>
                    Leading companies by market capitalization
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[120px]">Ticker</TableHead>
                        <TableHead>Company</TableHead>
                        <TableHead className="text-right">Price</TableHead>
                        <TableHead className="text-right">Change</TableHead>
                        <TableHead className="text-right">Market Weight</TableHead>
                        <TableHead className="text-right">P/E Ratio</TableHead>
                        <TableHead>Rating</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {isSectorLoading ? (
                        <TableRow>
                          <TableCell colSpan={7} className="text-center py-4">
                            Loading...
                          </TableCell>
                        </TableRow>
                      ) : topStocks[sectorKey]?.map((stock) => (
                        <TableRow key={stock.symbol}>
                          <TableCell className="font-medium">{stock.symbol}</TableCell>
                          <TableCell>{stock.name}</TableCell>
                          <TableCell className="text-right">${stock.stock_price.toFixed(2)}</TableCell>
                          <TableCell className={`text-right flex items-center justify-end gap-1 ${stock.percent_change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {stock.percent_change >= 0 ? <ArrowUp size={14} /> : <ArrowDown size={14} />}
                            {Math.abs(stock.percent_change).toFixed(2)}%
                          </TableCell>
                          <TableCell className="text-right">{(stock.market_weight * 100).toFixed(2)}%</TableCell>
                          <TableCell className="text-right">{stock.pe_ratio.toFixed(1)}x</TableCell>
                          <TableCell>
                            <Badge variant={
                              stock.rating === 'Strong Buy' ? 'default' :
                              stock.rating === 'Buy' ? 'secondary' :
                              stock.rating === 'Hold' ? 'outline' :
                              'destructive'
                            }>
                              {stock.rating}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* SWOT Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Compass className="h-5 w-5" />
                    SWOT Analysis: {sectors[sectorKey].name}
                  </CardTitle>
                  <CardDescription>
                    AI-powered strengths, weaknesses, opportunities, and threats analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="border-green-500/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-green-600">
                        Strengths
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <ul className="space-y-2 text-sm">
                        {swotAnalysis[sectorKey as keyof typeof swotAnalysis]?.strengths.map((item, i) => (
                          <li key={`strength-${i}`} className="flex items-start gap-2">
                            <div className="bg-green-500/10 p-1 rounded-full mt-0.5">
                              <Award className="h-3 w-3 text-green-600" />
                            </div>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="border-red-500/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-red-600">
                        Weaknesses
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <ul className="space-y-2 text-sm">
                        {swotAnalysis[sectorKey as keyof typeof swotAnalysis]?.weaknesses.map((item, i) => (
                          <li key={`weakness-${i}`} className="flex items-start gap-2">
                            <div className="bg-red-500/10 p-1 rounded-full mt-0.5">
                              <ArrowDownLeft className="h-3 w-3 text-red-600" />
                            </div>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="border-blue-500/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-blue-600">
                        Opportunities
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <ul className="space-y-2 text-sm">
                        {swotAnalysis[sectorKey as keyof typeof swotAnalysis]?.opportunities.map((item, i) => (
                          <li key={`opportunity-${i}`} className="flex items-start gap-2">
                            <div className="bg-blue-500/10 p-1 rounded-full mt-0.5">
                              <ArrowRightCircle className="h-3 w-3 text-blue-600" />
                            </div>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="border-amber-500/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-amber-600">
                        Threats
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <ul className="space-y-2 text-sm">
                        {swotAnalysis[sectorKey as keyof typeof swotAnalysis]?.threats.map((item, i) => (
                          <li key={`threat-${i}`} className="flex items-start gap-2">
                            <div className="bg-amber-500/10 p-1 rounded-full mt-0.5">
                              <Gauge className="h-3 w-3 text-amber-600" />
                            </div>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                </CardContent>
              </Card>

              {/* Industry Dependencies */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Industry Dependencies
                  </CardTitle>
                  <CardDescription>
                    Key interdependencies and external factors affecting sector performance
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {industryDependencies[sectorKey as keyof typeof industryDependencies]?.map((item, i) => (
                      <div key={`dependency-${i}`} className="p-4 border rounded-lg">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge variant={item.impact === 'Critical' ? 'destructive' : item.impact === 'High' ? 'default' : 'secondary'}>
                              {item.impact} Impact
                            </Badge>
                            <h3 className="font-semibold">{item.sector} Sector</h3>
                          </div>
                        </div>
                        <p className="text-sm mt-2 text-muted-foreground">
                          {item.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* ESG Ranking */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Globe className="h-5 w-5" />
                    ESG Leaders in {sectors[sectorKey].name}
                  </CardTitle>
                  <CardDescription>
                    Top performing companies in environmental, social, and governance criteria
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {esgRankings.map((company) => (
                      <div key={company.ticker} className="flex items-center gap-3 p-3 border rounded-lg">
                        <div className="flex-shrink-0 bg-primary/10 w-10 h-10 rounded-full flex items-center justify-center text-primary font-semibold">
                          {company.rank}
                        </div>
                        <div className="flex-grow">
                          <div className="flex justify-between items-center">
                            <div>
                              <h3 className="font-semibold">{company.company} <span className="text-muted-foreground font-normal">({company.ticker})</span></h3>
                              <div className="flex items-center gap-4 mt-1 text-xs">
                                <div>
                                  <span className="text-muted-foreground">Environment:</span> <span className="font-medium">{company.environment}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Social:</span> <span className="font-medium">{company.social}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Governance:</span> <span className="font-medium">{company.governance}</span>
                                </div>
                              </div>
                            </div>
                            <div className="ml-auto pl-4">
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div className="flex items-center gap-2 bg-green-500/10 px-3 py-1 rounded-full">
                                      <span className="text-sm font-semibold text-green-600">{company.score}</span>
                                      <span className="text-xs text-muted-foreground">/100</span>
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>ESG Score: {company.score}/100</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* AI Sector Outlook */}
              <Card className="border-primary/20 bg-primary/[0.03]">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-primary" />
                    QuantAI Sector Outlook
                  </CardTitle>
                </CardHeader>
                <CardContent className="prose prose-sm max-w-none">
                  <div>
                    <p>{aiOutlook.overview}</p>
                    <p>Key subsector trends include:</p>
                    <ul>
                      {aiOutlook.keyTrends.map((trend, index) => (
                        <li key={index}>{trend}</li>
                      ))}
                    </ul>
                    <p>{aiOutlook.capitalAllocation}</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </Layout>
  );
};

export default Sectors;