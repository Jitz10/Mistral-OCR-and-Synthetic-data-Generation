
import React from "react";
import Layout from "@/components/layout/Layout";
import MarketIndices from "@/components/dashboard/MarketIndices";
import TrendingStocks from "@/components/dashboard/TrendingStocks";
import SectorHeatmap from "@/components/dashboard/SectorHeatmap";
import AiInsights from "@/components/dashboard/AiInsights";
import AiChatbot from "@/components/shared/AiChatbot";

const Index = () => {
  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground">
              Welcome back to QuantAI, your AI-powered financial insights platform
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Market is open</p>
            <p className="text-lg font-medium">Apr 12, 2025 • 10:43 AM EST</p>
          </div>
        </div>

        <MarketIndices />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrendingStocks />
          <AiInsights />
        </div>
        <SectorHeatmap />
      </div>
      <AiChatbot />
    </Layout>
  );
};

export default Index;
