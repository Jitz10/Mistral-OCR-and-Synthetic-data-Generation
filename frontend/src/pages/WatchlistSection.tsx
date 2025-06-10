import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { FileEdit, Plus, Trash2, ArrowUp, ArrowDown } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';

// Initial watchlist data
const initialWatchlist = [
  { ticker: 'NVDA', name: 'NVIDIA Corporation', price: 870.39, change: 2.87, marketCap: 2140, peRatio: 68.4, aiRec: 'Strong Buy', notes: 'AI leader with strong growth' },
  { ticker: 'META', name: 'Meta Platforms, Inc.', price: 486.18, change: 1.32, marketCap: 1245, peRatio: 26.2, aiRec: 'Buy', notes: 'Rebound in ad spend' },
  { ticker: 'AMD', name: 'Advanced Micro Devices', price: 152.37, change: 1.87, marketCap: 246, peRatio: 88.5, aiRec: 'Buy', notes: 'Growing server market share' },
  { ticker: 'PYPL', name: 'PayPal Holdings, Inc.', price: 64.28, change: -0.85, marketCap: 68.5, peRatio: 17.3, aiRec: 'Hold', notes: 'Competition concerns' },
  { ticker: 'DIS', name: 'Walt Disney Company', price: 114.52, change: 0.42, marketCap: 209.7, peRatio: 78.4, aiRec: 'Buy', notes: 'Streaming and parks recovery' },
];

const WatchlistSection = () => {
  const { toast } = useToast();
  const [watchlist, setWatchlist] = useState(initialWatchlist);
  const [isAddStockOpen, setIsAddStockOpen] = useState(false);
  const [newStock, setNewStock] = useState({
    ticker: '',
    name: '',
    price: '',
    change: '',
    marketCap: '',
    peRatio: '',
    aiRec: 'Buy',
    notes: '',
  });

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

  const handleDeleteStock = (ticker: string) => {
    setWatchlist(watchlist.filter(stock => stock.ticker !== ticker));
    toast({
      title: "Stock Removed",
      description: `${ticker} has been removed from your watchlist.`,
    });
  };

  const handleAddStock = () => {
    if (!newStock.ticker || !newStock.name || !newStock.price) {
      toast({
        title: "Error",
        description: "Please fill in at least Ticker, Name, and Price.",
        variant: "destructive",
      });
      return;
    }

    const stockToAdd = {
      ticker: newStock.ticker.toUpperCase(),
      name: newStock.name,
      price: parseFloat(newStock.price) || 0,
      change: parseFloat(newStock.change) || 0,
      marketCap: parseFloat(newStock.marketCap) || 0,
      peRatio: parseFloat(newStock.peRatio) || 0,
      aiRec: newStock.aiRec,
      notes: newStock.notes || 'No notes provided',
    };

    setWatchlist([...watchlist, stockToAdd]);
    setNewStock({
      ticker: '',
      name: '',
      price: '',
      change: '',
      marketCap: '',
      peRatio: '',
      aiRec: 'Buy',
      notes: '',
    });
    setIsAddStockOpen(false);
    toast({
      title: "Stock Added",
      description: `${stockToAdd.ticker} has been added to your watchlist.`,
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle>Watchlist</CardTitle>
          <Button size="sm" onClick={() => setIsAddStockOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Stock
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">Ticker</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead className="text-right">Price</TableHead>
                  <TableHead className="text-right">Change</TableHead>
                  <TableHead>AI Rating</TableHead>
                  <TableHead className="w-[100px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {watchlist.map(stock => (
                  <TableRow key={stock.ticker}>
                    <TableCell className="font-medium">{stock.ticker}</TableCell>
                    <TableCell className="max-w-[200px] truncate">{stock.name}</TableCell>
                    <TableCell className="text-right">${stock.price.toFixed(2)}</TableCell>
                    <TableCell className={`text-right ${getChangeColor(stock.change)}`}>
                      <div className="flex items-center justify-end gap-1">
                        {getChangeIcon(stock.change)}
                        <span>{Math.abs(stock.change).toFixed(2)}%</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      {getRecommendationBadge(stock.aiRec)}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => handleDeleteStock(stock.ticker)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Watchlist Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {watchlist.map(stock => (
          <Card key={`${stock.ticker}-note`} className="overflow-hidden">
            <CardHeader className="bg-muted/50 pb-2">
              <div className="flex justify-between items-center">
                <CardTitle className="text-base flex items-center gap-2">
                  <span>{stock.ticker}</span>
                  <span className="text-sm font-normal text-muted-foreground">{stock.name}</span>
                </CardTitle>
                <div className="flex gap-1">
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <FileEdit className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-3">
              <div className="text-sm">{stock.notes}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Add Stock Dialog */}
      <Dialog open={isAddStockOpen} onOpenChange={setIsAddStockOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add New Stock</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="ticker" className="text-right">Ticker</label>
              <Input
                id="ticker"
                value={newStock.ticker}
                onChange={(e) => setNewStock({ ...newStock, ticker: e.target.value })}
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="name" className="text-right">Name</label>
              <Input
                id="name"
                value={newStock.name}
                onChange={(e) => setNewStock({ ...newStock, name: e.target.value })}
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="price" className="text-right">Price</label>
              <Input
                id="price"
                type="number"
                value={newStock.price}
                onChange={(e) => setNewStock({ ...newStock, price: e.target.value })}
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="change" className="text-right">Change (%)</label>
              <Input
                id="change"
                type="number"
                value={newStock.change}
                onChange={(e) => setNewStock({ ...newStock, change: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="marketCap" className="text-right">Market Cap</label>
              <Input
                id="marketCap"
                type="number"
                value={newStock.marketCap}
                onChange={(e) => setNewStock({ ...newStock, marketCap: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="peRatio" className="text-right">P/E Ratio</label>
              <Input
                id="peRatio"
                type="number"
                value={newStock.peRatio}
                onChange={(e) => setNewStock({ ...newStock, peRatio: e.target.value })}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="aiRec" className="text-right">AI Rating</label>
              <select
                id="aiRec"
                value={newStock.aiRec}
                onChange={(e) => setNewStock({ ...newStock, aiRec: e.target.value })}
                className="col-span-3 border rounded p-2"
              >
                <option value="Strong Buy">Strong Buy</option>
                <option value="Buy">Buy</option>
                <option value="Hold">Hold</option>
                <option value="Sell">Sell</option>
              </select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="notes" className="text-right">Notes</label>
              <Input
                id="notes"
                value={newStock.notes}
                onChange={(e) => setNewStock({ ...newStock, notes: e.target.value })}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddStockOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddStock}>Add Stock</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default WatchlistSection;