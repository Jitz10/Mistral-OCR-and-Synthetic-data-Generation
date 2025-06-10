import { MarketOverview } from "../components/dashboard/MarketOverview";
import TrendingStocks from "@/components/dashboard/TrendingStocks";
import SectorHeatmap from "@/components/dashboard/SectorHeatmap";
import AiInsights from "@/components/dashboard/AiInsights";
import { AIChatbot } from "../components/ui/ai-chatbot";
import Layout from "@/components/layout/Layout";

const MarketInsights = () => {
  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">MarketInsights</h1>
          <p className="text-muted-foreground">
            Market overview and key insights for {new Date().toLocaleDateString()}
          </p>
        </div>

        <MarketOverview />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <TrendingStocks />
          </div>
          <div className="lg:col-span-1">
            <AiInsights />
          </div>
        </div>

        <div className="mt-6">
          <SectorHeatmap />
        </div>
      </div>
    </Layout>
  );
};

export default MarketInsights;