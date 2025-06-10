
import React, { useState } from 'react';
import Layout from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from '@/components/ui/separator';
import {
  BarChart, LineChart as RechartsLineChart, AreaChart,
  Bar, Line, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip as RechartsTooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  LineChart, Activity, ArrowDown, ArrowRight, ArrowUp, ChevronDown,
  ChevronRight, CircleDashed, Columns3, Download, Filter, FileDown,
  LineChart as LineChartIcon, Laptop, Pin, Plus, Save, Settings2,
  SlidersHorizontal, Smartphone, Tablet, Trash2, Zap, Search,
  BarChart4, MoreHorizontal, Sparkles
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
  PaginationEllipsis,
} from "@/components/ui/pagination";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from "@/components/ui/chart";

const Screener = () => {
  const [view, setView] = useState('table');
  const [selectedMarketCap, setSelectedMarketCap] = useState(['small', 'mid', 'large']);
  const [selectedSectors, setSelectedSectors] = useState(['all']);
  const [priceRange, setPriceRange] = useState([0, 1000]);
  const [filters, setFilters] = useState<any>({});

  const data = [
    { name: 'AAPL', price: 150.23, change: 2.5, marketCap: 2500, pe: 25.3, dividend: 0.6, volume: 32.5, sector: 'Technology', sparkline: [150, 152, 148, 155, 153, 158, 160] },
    { name: 'MSFT', price: 290.10, change: 1.8, marketCap: 2200, pe: 32.1, dividend: 0.8, volume: 28.2, sector: 'Technology', sparkline: [290, 295, 288, 300, 297, 305, 310] },
    { name: 'GOOGL', price: 135.15, change: 0.7, marketCap: 1800, pe: 27.5, dividend: 0, volume: 15.7, sector: 'Technology', sparkline: [135, 137, 132, 140, 138, 142, 145] },
    { name: 'AMZN', price: 128.90, change: 0.5, marketCap: 1300, pe: 60.2, dividend: 0, volume: 22.3, sector: 'Consumer Cyclical', sparkline: [128, 130, 125, 133, 130, 135, 138] },
    { name: 'META', price: 320.54, change: 3.2, marketCap: 820, pe: 29.4, dividend: 0, volume: 18.6, sector: 'Technology', sparkline: [320, 325, 318, 330, 328, 335, 340] },
    { name: 'TSLA', price: 258.73, change: -1.2, marketCap: 800, pe: 75.3, dividend: 0, volume: 30.1, sector: 'Consumer Cyclical', sparkline: [258, 255, 262, 250, 253, 245, 240] },
    { name: 'V', price: 241.20, change: 0.3, marketCap: 500, pe: 35.1, dividend: 0.7, volume: 10.2, sector: 'Financial Services', sparkline: [241, 243, 238, 245, 243, 247, 250] },
    { name: 'JNJ', price: 160.09, change: -0.8, marketCap: 420, pe: 22.5, dividend: 2.8, volume: 7.8, sector: 'Healthcare', sparkline: [160, 157, 163, 155, 158, 153, 150] },
    { name: 'JPM', price: 138.45, change: 1.5, marketCap: 410, pe: 11.2, dividend: 2.5, volume: 9.5, sector: 'Financial Services', sparkline: [138, 141, 135, 145, 143, 148, 150] },
    { name: 'PG', price: 155.32, change: 0.2, marketCap: 380, pe: 27.8, dividend: 2.3, volume: 6.3, sector: 'Consumer Defensive', sparkline: [155, 156, 153, 158, 157, 160, 162] },
  ];

  const sectors = ['All Sectors', 'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Consumer Defensive', 'Industrials', 'Basic Materials', 'Energy', 'Utilities', 'Real Estate', 'Communication Services'];

  const renderSparklineChart = (data: number[]) => {
    return (
      <ChartContainer config={{}} className="h-8 w-16">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data.map((value, i) => ({ value, i }))} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="sparklineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <ChartTooltip
              content={<ChartTooltipContent />}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3B82F6"
              fill="url(#sparklineGradient)"
              dot={false}
              isAnimationActive={false}
              strokeWidth={1.5}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    );
  };

  // Filter handlers
  const handleMarketCapChange = (value: string) => {
    if (value === 'all') {
      setSelectedMarketCap(['small', 'mid', 'large']);
    } else {
      if (selectedMarketCap.includes(value)) {
        setSelectedMarketCap(selectedMarketCap.filter(item => item !== value));
      } else {
        setSelectedMarketCap([...selectedMarketCap, value]);
      }
    }
  };

  const handleSectorChange = (value: string) => {
    if (value === 'all') {
      setSelectedSectors(['all']);
    } else {
      if (selectedSectors.includes('all')) {
        setSelectedSectors([value]);
      } else if (selectedSectors.includes(value)) {
        if (selectedSectors.length === 1) {
          setSelectedSectors(['all']);
        } else {
          setSelectedSectors(selectedSectors.filter(sector => sector !== value));
        }
      } else {
        setSelectedSectors([...selectedSectors.filter(sector => sector !== 'all'), value]);
      }
    }
  };

  return (
    <Layout>
      <h1>Screener</h1>
      <div className="container py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Stock Screener</h1>
          <div className="flex items-center gap-2">
            <Button variant="outline" className="flex items-center gap-2">
              <Save size={16} />
              Save
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <Download size={16} />
              Export
            </Button>
          </div>
        </div>

        <Card className="mb-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl">Filters</CardTitle>
            <CardDescription>Customize your stock screening criteria</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Label className="mb-2 block">Market Cap</Label>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant={selectedMarketCap.length === 3 ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleMarketCapChange('all')}
                  >
                    All
                  </Button>
                  <Button
                    variant={selectedMarketCap.includes('small') ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleMarketCapChange('small')}
                  >
                    Small
                  </Button>
                  <Button
                    variant={selectedMarketCap.includes('mid') ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleMarketCapChange('mid')}
                  >
                    Mid
                  </Button>
                  <Button
                    variant={selectedMarketCap.includes('large') ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleMarketCapChange('large')}
                  >
                    Large
                  </Button>
                </div>
              </div>
              
              <div>
                <Label className="mb-2 block">Sector</Label>
                <Select>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select sector" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel>Sectors</SelectLabel>
                      {sectors.map((sector) => (
                        <SelectItem key={sector} value={sector.toLowerCase()}>{sector}</SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label className="mb-2 block">Price Range ($)</Label>
                <div className="pt-4 px-2">
                  <Slider
                    defaultValue={[0, 1000]}
                    max={1000}
                    step={10}
                    onValueChange={setPriceRange}
                  />
                  <div className="flex justify-between mt-2 text-sm text-muted-foreground">
                    <span>${priceRange[0]}</span>
                    <span>${priceRange[1]}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <Label className="mb-2 block">P/E Ratio</Label>
                <div className="pt-4 px-2">
                  <Slider
                    defaultValue={[0, 100]}
                    max={100}
                    step={1}
                  />
                  <div className="flex justify-between mt-2 text-sm text-muted-foreground">
                    <span>0</span>
                    <span>100+</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-3 mt-6">
              <Button size="sm" className="flex items-center gap-1.5">
                <Filter size={14} />
                Dividend Yield {'>'} 0%
                <ChevronDown size={14} />
              </Button>
              <Button size="sm" className="flex items-center gap-1.5">
                <Filter size={14} />
                Add Filter
                <Plus size={14} />
              </Button>
              <Button variant="outline" size="sm">
                Reset All
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search results..."
                className="pl-9 w-[250px]"
              />
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="flex items-center gap-1">
                  <Columns3 size={16} />
                  Columns
                  <ChevronDown size={14} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>Toggle Columns</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuCheckboxItem checked={true}>
                  Symbol
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Price
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Change %
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Market Cap
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  P/E Ratio
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Dividend %
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Volume (M)
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Sector
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem checked={true}>
                  Chart
                </DropdownMenuCheckboxItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          
          <div className="flex items-center gap-3">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="outline">
                    <BarChart4 size={18} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>View as Bar Chart</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="outline">
                    <LineChart size={18} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>View as Line Chart</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="outline">
                    <MoreHorizontal size={18} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>More Options</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" variant="default">
                    <Sparkles size={16} className="mr-2" />
                    AI Screen
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Generate AI-powered screening criteria</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>

        <Card>
          <ScrollArea className="h-[600px]">
            <Table>
              <TableHeader className="sticky top-0 bg-background">
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Price ($)</TableHead>
                  <TableHead>Change (%)</TableHead>
                  <TableHead className="text-right">Market Cap ($B)</TableHead>
                  <TableHead className="text-right">P/E Ratio</TableHead>
                  <TableHead className="text-right">Dividend (%)</TableHead>
                  <TableHead className="text-right">Volume (M)</TableHead>
                  <TableHead>Sector</TableHead>
                  <TableHead className="w-[100px]">Chart</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.map((stock) => (
                  <TableRow key={stock.name}>
                    <TableCell className="font-medium">{stock.name}</TableCell>
                    <TableCell>${stock.price.toFixed(2)}</TableCell>
                    <TableCell>
                      <span className={stock.change > 0 ? "text-green-500 flex items-center" : "text-red-500 flex items-center"}>
                        {stock.change > 0 ? <ArrowUp size={14} className="mr-1"/> : <ArrowDown size={14} className="mr-1" />}
                        {Math.abs(stock.change)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-right">{stock.marketCap.toFixed(1)}</TableCell>
                    <TableCell className="text-right">{stock.pe.toFixed(1)}</TableCell>
                    <TableCell className="text-right">{stock.dividend.toFixed(1)}</TableCell>
                    <TableCell className="text-right">{stock.volume.toFixed(1)}</TableCell>
                    <TableCell>{stock.sector}</TableCell>
                    <TableCell>{renderSparklineChart(stock.sparkline)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
          
          <div className="p-4 flex items-center justify-between border-t">
            <div className="text-sm text-muted-foreground">
              Showing 10 of 1,248 results
            </div>
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious href="#" />
                </PaginationItem>
                <PaginationItem>
                  <PaginationLink href="#" isActive>1</PaginationLink>
                </PaginationItem>
                <PaginationItem>
                  <PaginationLink href="#">2</PaginationLink>
                </PaginationItem>
                <PaginationItem>
                  <PaginationLink href="#">3</PaginationLink>
                </PaginationItem>
                <PaginationItem>
                  <PaginationEllipsis />
                </PaginationItem>
                <PaginationItem>
                  <PaginationNext href="#" />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        </Card>

        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles size={18} />
                AI-Generated Stock Recommendations
              </div>
              <Button variant="outline" size="sm">View All</Button>
            </CardTitle>
            <CardDescription>Personalized recommendations based on your screening criteria</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {data.slice(0, 6).map((stock) => (
                  <Card key={stock.name} className="overflow-hidden">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-lg font-semibold">{stock.name}</p>
                          <p className="text-sm text-muted-foreground">{stock.sector}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-medium">${stock.price.toFixed(2)}</p>
                          <span className={stock.change > 0 ? "text-green-500 text-sm" : "text-red-500 text-sm"}>
                            {stock.change > 0 ? "+" : ""}{stock.change}%
                          </span>
                        </div>
                      </div>
                      <div className="mt-3">
                        {renderSparklineChart(stock.sparkline)}
                      </div>
                      <p className="mt-3 text-xs text-muted-foreground">Strong fundamentals with consistent dividend growth</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
        
        <div className="mt-6 text-center">
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious href="#" />
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#" isActive>1</PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#">2</PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationLink href="#">3</PaginationLink>
              </PaginationItem>
              <PaginationItem>
                <PaginationEllipsis />
              </PaginationItem>
              <PaginationItem>
                <PaginationNext href="#" />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      </div>
    </Layout>
  );
};

export default Screener;
