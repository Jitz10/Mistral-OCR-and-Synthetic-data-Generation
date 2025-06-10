import React, { useState } from 'react';
import Layout from '@/components/layout/Layout';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import TrendingNews from './TrendingNews';
import MarketForecasts from './MarketForecasts';
import InvestmentAnalysisAndSentiment from './InvestmentAnalysisAndSentiment';

const News = () => {
  const [activeTab, setActiveTab] = useState('trending');

  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">News & Forecasts</h1>
            <p className="text-muted-foreground">
              AI-powered financial news analysis and market forecasts
            </p>
          </div>
          <div className="flex items-center gap-2"></div>
        </div>

        <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="trending">Trending News</TabsTrigger>
            <TabsTrigger value="forecasts">Market Forecasts</TabsTrigger>
            <TabsTrigger value="analysis">Investment Analysis</TabsTrigger>
            <TabsTrigger value="sentiment">Sentiment Tracker</TabsTrigger>
          </TabsList>

          <TabsContent value="trending" className="space-y-4">
            <TrendingNews />
          </TabsContent>
          <TabsContent value="forecasts" className="space-y-4">
            <MarketForecasts />
          </TabsContent>
          <TabsContent value="analysis" className="space-y-4">
            <InvestmentAnalysisAndSentiment tab="analysis" />
          </TabsContent>
          <TabsContent value="sentiment" className="space-y-4">
            <InvestmentAnalysisAndSentiment tab="sentiment" />
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
};

export default News;