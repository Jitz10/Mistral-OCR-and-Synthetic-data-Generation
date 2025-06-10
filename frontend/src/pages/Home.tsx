import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '@/components/layout/Layout';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
// import UploadFileBox from '@/UploadFileBox';

const Home = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // You can add logic here to determine if this is a stock or sector search
      navigate(`/sectors?query=${encodeURIComponent(searchQuery)}`);
    }
  };

  const trendingStocks = [
    { name: 'Apple Inc.', symbol: 'AAPL', change: '+2.5%' },
    { name: 'Microsoft', symbol: 'MSFT', change: '+1.8%' },
    { name: 'Amazon', symbol: 'AMZN', change: '+0.5%' },
    { name: 'Tesla', symbol: 'TSLA', change: '-1.2%' },
    { name: 'Google', symbol: 'GOOGL', change: '+0.7%' }
  ];

  const marketSectors = [
    { id: 'basic-materials', name: 'Basic Materials' },
    { id: 'communication-services', name: 'Communication Services' },
    { id: 'consumer-cyclical', name: 'Consumer Cyclical' },
    { id: 'consumer-defensive', name: 'Consumer Defensive' },
    { id: 'energy', name: 'Energy' },
    { id: 'financial-services', name: 'Financial Services' },
    { id: 'healthcare', name: 'Healthcare' },
    { id: 'industrials', name: 'Industrials' },
    { id: 'real-estate', name: 'Real Estate' },
    { id: 'technology', name: 'Technology' },
    { id: 'utilities', name: 'Utilities' }
  ];

  return (
    <Layout>
      <section className="relative h-[500px] flex items-center justify-center bg-gradient-to-br from-primary/10 to-secondary/5">
        <div className="container max-w-4xl mx-auto text-center px-4">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            Stock analysis and screening tool for investors
          </h1>
          <p className="text-xl text-muted-foreground mb-8">
            Research stocks, compare companies, and make informed investment decisions
          </p>
          <form onSubmit={handleSearch} className="relative max-w-xl mx-auto">
            <Search className="absolute left-4 top-3.5 h-5 w-5 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search for a sector..."
              className="pl-12 py-6 text-lg rounded-full"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <Button 
              type="submit" 
              className="absolute right-1.5 top-1.5 rounded-full"
            >
              Search
            </Button>
          </form>
        </div>
      </section>

      <section className="py-12 container">
        <div>
          <h2 className="text-2xl font-bold mb-6">Browse by Sector</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {marketSectors.map((sector) => (
              <Button 
                key={sector.id} 
                variant="secondary" 
                className="h-auto py-3"
                onClick={() => navigate(`/sectors?sector=${encodeURIComponent(sector.id)}`)}
              >
                {sector.name}
              </Button>
            ))}
          </div>
        </div>

        {/* <div className="mt-10">
          <UploadFileBox />
        </div> */}

        <div className="mt-10">
          <h2 className="text-2xl font-bold mb-6">Trending Stocks</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {trendingStocks.map((stock) => (
              <Button 
                key={stock.symbol} 
                variant="outline" 
                className="h-auto py-3 flex flex-col items-start"
                onClick={() => navigate(`/stock/${stock.symbol}`)}
              >
                <span className="text-lg font-semibold">{stock.symbol}</span>
                <span className="text-sm text-muted-foreground">{stock.name}</span>
                <span className={`text-sm ${stock.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
                  {stock.change}
                </span>
              </Button>
            ))}
          </div>
        </div>
      </section>
    </Layout>
  );
};

export default Home;
