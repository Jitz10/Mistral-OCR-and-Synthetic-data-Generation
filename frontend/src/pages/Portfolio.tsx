// src/pages/Portfolio.tsx
import React, { useState } from 'react';
import Layout from '@/components/layout/Layout';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Plus, Share2 } from 'lucide-react';
import PortfolioSection from './PortfolioSection';
import WatchlistSection from './WatchlistSection';
import AIInsightsSection from './AIInsightsSection';
import AddInvestmentForm from '@/components/AddInvestmentForm';
import { useToast } from '@/hooks/use-toast';

const initialPortfolioHoldings = [
  { ticker: 'AAPL', name: 'Apple Inc.', shares: 25, avgCost: 155.42, currentPrice: 187.68, marketValue: 4692, pl: 805.5, plPercent: 20.7, aiRec: 'Buy', sector: 'Technology' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', shares: 12, avgCost: 342.15, currentPrice: 405.12, marketValue: 4861.44, pl: 755.64, plPercent: 18.4, aiRec: 'Buy', sector: 'Technology' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', shares: 18, avgCost: 145.22, currentPrice: 173.45, marketValue: 3122.1, pl: 508.14, plPercent: 19.4, aiRec: 'Buy', sector: 'Consumer Cyclical' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', shares: 20, avgCost: 128.75, currentPrice: 151.35, marketValue: 3027, pl: 452, plPercent: 17.6, aiRec: 'Buy', sector: 'Communication Services' },
  { ticker: 'TSLA', name: 'Tesla, Inc.', shares: 15, avgCost: 312.45, currentPrice: 273.54, marketValue: 4103.1, pl: -584.65, plPercent: -12.5, aiRec: 'Hold', sector: 'Consumer Cyclical' },
  { ticker: 'JPM', name: 'JPMorgan Chase & Co.', shares: 10, avgCost: 172.34, currentPrice: 197.45, marketValue: 1974.5, pl: 251.1, plPercent: 14.6, aiRec: 'Buy', sector: 'Financial Services' },
  { ticker: 'JNJ', name: 'Johnson & Johnson', shares: 12, avgCost: 165.22, currentPrice: 154.78, marketValue: 1857.36, pl: -124.32, plPercent: -6.3, aiRec: 'Hold', sector: 'Healthcare' },
  { ticker: 'V', name: 'Visa Inc.', shares: 8, avgCost: 222.45, currentPrice: 248.32, marketValue: 1986.56, pl: 206.96, plPercent: 11.6, aiRec: 'Buy', sector: 'Financial Services' },
];

const Portfolio = () => {
  const [activeTab, setActiveTab] = useState('portfolio');
  const [showForm, setShowForm] = useState(false);
  const [portfolioHoldings, setPortfolioHoldings] = useState(initialPortfolioHoldings);
  const { toast } = useToast();

  const handleAddStock = (stock: typeof initialPortfolioHoldings[0]) => {
    setPortfolioHoldings([...portfolioHoldings, stock]);
    setShowForm(false);
  };

  const exportToCSV = () => {
    if (portfolioHoldings.length === 0) {
      toast({
        title: 'No Data to Export',
        description: 'Your portfolio is empty.',
        variant: 'destructive',
      });
      return;
    }

    // Define CSV headers
    const headers = [
      'Ticker',
      'Name',
      'Shares',
      'Average Cost',
      'Current Price',
      'Market Value',
      'P/L',
      'P/L %',
      'AI Recommendation',
      'Sector',
    ];

    // Convert holdings to CSV rows
    const rows = portfolioHoldings.map(holding => [
      holding.ticker,
      holding.name,
      holding.shares.toString(),
      holding.avgCost.toFixed(2),
      holding.currentPrice.toFixed(2),
      holding.marketValue.toFixed(2),
      holding.pl.toFixed(2),
      holding.plPercent.toFixed(1),
      holding.aiRec,
      holding.sector,
    ]);

    // Combine headers and rows
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(',')),
    ].join('\n');

    // Create a Blob for download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `portfolio_export_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: 'Export Successful',
      description: 'Your portfolio data has been exported as a CSV file.',
    });
  };

  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">My Portfolio</h1>
            <p className="text-muted-foreground">
              Track your investments and watch list with AI-powered insights
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={exportToCSV}>
              <Share2 className="mr-2 h-4 w-4" />
              Export
            </Button>
            <Button size="sm" onClick={() => setShowForm(true)} disabled={showForm}>
              <Plus className="mr-2 h-4 w-4" />
              Add Investment
            </Button>
          </div>
        </div>

        {showForm && (
          <AddInvestmentForm
            onAddStock={handleAddStock}
            onCancel={() => setShowForm(false)}
          />
        )}

        {!showForm && (
          <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
              <TabsTrigger value="watchlist">Watchlist</TabsTrigger>
              <TabsTrigger value="insights">AI Insights</TabsTrigger>
            </TabsList>

            <TabsContent value="portfolio" className="space-y-6">
              <PortfolioSection
                portfolioHoldings={portfolioHoldings}
                setPortfolioHoldings={setPortfolioHoldings}
              />
            </TabsContent>
            <TabsContent value="watchlist" className="space-y-6">
              <WatchlistSection />
            </TabsContent>
            <TabsContent value="insights" className="space-y-6">
              <AIInsightsSection />
            </TabsContent>
          </Tabs>
        )}
      </div>
    </Layout>
  );
};

export default Portfolio;