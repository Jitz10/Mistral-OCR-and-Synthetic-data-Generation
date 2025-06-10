import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Calendar, FileText, Globe, Bookmark as BookmarkIcon, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { Skeleton } from '@/components/ui/skeleton';

interface Headline {
  title: string;
  description: string | null;
  image: string | null;
  published_at: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  source: string;
  url: string;
}

const TrendingNews = () => {
  const { toast } = useToast();

  const { data: newsData, isLoading, error } = useQuery({
    queryKey: ['stockMarketHeadlines'],
    queryFn: async () => {
      const response = await fetch('http://127.0.0.1:5000/api/stock-market-headlines', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const result = await response.json();
      if (result.status !== 'success') {
        throw new Error('API returned unsuccessful status');
      }

      return result.headlines as Headline[];
    },
  });

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <Badge className="bg-green-500 text-white">Positive</Badge>;
      case 'negative':
        return <Badge className="bg-red-500 text-white">Negative</Badge>;
      default:
        return <Badge className="bg-blue-500 text-white">Neutral</Badge>;
    }
  };

  const getCategory = (title: string): string => {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('stock market') || lowerTitle.includes('dow') || lowerTitle.includes('s&p 500')) return 'Markets';
    if (lowerTitle.includes('tariff') || lowerTitle.includes('trade war')) return 'Trade';
    if (lowerTitle.includes('steel') || lowerTitle.includes('manufacturing')) return 'Industry';
    if (lowerTitle.includes('earnings') || lowerTitle.includes('execs')) return 'Earnings';
    if (lowerTitle.includes('consumer sentiment') || lowerTitle.includes('inflation')) return 'Economy';
    if (lowerTitle.includes('electric') || lowerTitle.includes('gm')) return 'Automotive';
    if (lowerTitle.includes('treasury') || lowerTitle.includes('dollar')) return 'Bonds';
    if (lowerTitle.includes('mortgage') || lowerTitle.includes('rates')) return 'Housing';
    if (lowerTitle.includes('bitcoin') || lowerTitle.includes('stablecoin')) return 'Crypto';
    return 'General';
  };

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[...Array(6)].map((_, index) => (
          <Card key={index}>
            <CardHeader>
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-4 w-full mt-4" />
              <Skeleton className="h-4 w-2/3 mt-2" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">Error loading news. Please try again later.</p>
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {newsData?.map((news: Headline, index: number) => (
        <Card key={index} className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg line-clamp-2">{news.title}</CardTitle>
              <span className={`px-2 py-1 rounded-full text-xs ${
                news.sentiment === 'positive' ? 'bg-green-100 text-green-800' :
                news.sentiment === 'negative' ? 'bg-red-100 text-red-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {news.sentiment}
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>{news.source}</span>
              <span>•</span>
              <span>{new Date(news.published_at).toLocaleDateString()}</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video mb-4">
              <img
                src={news.image || '/news.jpeg'}
                alt={news.title}
                className="rounded-lg object-cover w-full h-full"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = '/news.jpeg';
                }}
              />
            </div>
            <p className="text-sm text-muted-foreground line-clamp-3">{news.description || 'No description available'}</p>
            <a
              href={news.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-primary hover:underline mt-4"
            >
              Read more
              <ArrowUpRight size={16} />
            </a>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default TrendingNews;