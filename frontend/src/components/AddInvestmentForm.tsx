// src/components/AddInvestmentForm.tsx
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';

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

interface AddInvestmentFormProps {
  onAddStock: (stock: Stock) => void;
  onCancel: () => void;
}

const AddInvestmentForm = ({ onAddStock, onCancel }: AddInvestmentFormProps) => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    ticker: '',
    name: '',
    shares: '',
    avgCost: '',
    currentPrice: '',
    sector: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const shares = parseFloat(formData.shares);
    const avgCost = parseFloat(formData.avgCost);
    const currentPrice = parseFloat(formData.currentPrice);

    if (!formData.ticker || !formData.name || isNaN(shares) || isNaN(avgCost) || isNaN(currentPrice) || !formData.sector) {
      toast({
        title: 'Invalid Input',
        description: 'Please fill all fields with valid data.',
        variant: 'destructive',
      });
      return;
    }

    const marketValue = shares * currentPrice;
    const pl = (currentPrice - avgCost) * shares;
    const plPercent = ((currentPrice - avgCost) / avgCost) * 100;

    const newStock: Stock = {
      ticker: formData.ticker.toUpperCase(),
      name: formData.name,
      shares,
      avgCost,
      currentPrice,
      marketValue,
      pl,
      plPercent,
      aiRec: 'Buy', // Default; could be dynamic
      sector: formData.sector,
    };

    onAddStock(newStock);
    toast({
      title: 'Stock Added',
      description: `${formData.ticker} has been added to your portfolio.`,
    });

    // Reset form
    setFormData({
      ticker: '',
      name: '',
      shares: '',
      avgCost: '',
      currentPrice: '',
      sector: '',
    });
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Add New Investment</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="ticker">Ticker</Label>
            <Input
              id="ticker"
              name="ticker"
              value={formData.ticker}
              onChange={handleChange}
              placeholder="e.g., AAPL"
            />
          </div>
          <div>
            <Label htmlFor="name">Company Name</Label>
            <Input
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="e.g., Apple Inc."
            />
          </div>
          <div>
            <Label htmlFor="shares">Shares</Label>
            <Input
              id="shares"
              name="shares"
              type="number"
              value={formData.shares}
              onChange={handleChange}
              placeholder="e.g., 10"
            />
          </div>
          <div>
            <Label htmlFor="avgCost">Average Cost ($)</Label>
            <Input
              id="avgCost"
              name="avgCost"
              type="number"
              step="0.01"
              value={formData.avgCost}
              onChange={handleChange}
              placeholder="e.g., 150.00"
            />
          </div>
          <div>
            <Label htmlFor="currentPrice">Current Price ($)</Label>
            <Input
              id="currentPrice"
              name="currentPrice"
              type="number"
              step="0.01"
              value={formData.currentPrice}
              onChange={handleChange}
              placeholder="e.g., 180.00"
            />
          </div>
          <div>
            <Label htmlFor="sector">Sector</Label>
            <Input
              id="sector"
              name="sector"
              value={formData.sector}
              onChange={handleChange}
              placeholder="e.g., Technology"
            />
          </div>
          <div className="flex gap-2">
            <Button type="submit">Add Stock</Button>
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

export default AddInvestmentForm;