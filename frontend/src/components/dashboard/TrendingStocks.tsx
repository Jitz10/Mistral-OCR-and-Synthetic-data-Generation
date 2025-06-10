
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, ResponsiveContainer } from "recharts";
import { ArrowDown, ArrowUp, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

// Mock data for trending stocks
const trendingStocks = [
  {
    symbol: "AAPL",
    name: "Apple Inc.",
    price: "$198.15",
    change: "+4.00%",
    isPositive: true,
    aiInsight: "Strong buy based on product cycle and services growth",
    data: [
      { value: 190 },
      { value: 192 },
      { value: 191 },
      { value: 195 },
      { value: 196 },
      { value: 197 },
      { value: 198 },
    ],
  },
  {
    symbol: "MSFT",
    name: "Microsoft Corp.",
    price: "$388.45",
    change: "+1.74%",
    isPositive: true,
    aiInsight: "Outperform due to AI integration and cloud momentum",
    data: [
      { value: 380 },
      { value: 382 },
      { value: 384 },
      { value: 386 },
      { value: 387 },
      { value: 388 },
      { value: 388 },
    ],
  },
  {
    symbol: "TSLA",
    name: "Tesla, Inc.",
    price: "$252.31",
    change: "-0.08%",
    isPositive: false,
    aiInsight: "Hold with caution on production challenges",
    data: [
      { value: 260 },
      { value: 258 },
      { value: 256 },
      { value: 254 },
      { value: 253 },
      { value: 252 },
      { value: 252 },
    ],
  },
  {
    symbol: "NVDA",
    name: "NVIDIA Corp.",
    price: "$110.93",
    change: "+2.87%",
    isPositive: true,
    aiInsight: "Strong buy on AI chip demand and data center growth",
    data: [
      { value: 105 },
      { value: 106 },
      { value: 107 },
      { value: 108 },
      { value: 109 },
      { value: 110 },
      { value: 110 },
    ],
  },
  {
    symbol: "GOOGL",
    name: "Alphabet Inc.",
    price: "$157.14",
    change: "+2.75%",
    isPositive: true,
    aiInsight: "Buy on strong advertising recovery and AI-driven cloud expansion",
    data: [
      { value: 150 },
      { value: 152 },
      { value: 153 },
      { value: 155 },
      { value: 156 },
      { value: 157 },
      { value: 157 },
    ],
  },
  {
    symbol: "AMZN",
    name: "Amazon.com Inc.",
    price: "$184.87",
    change: "+2.01%",
    isPositive: true,
    aiInsight: "Buy on AWS growth and e-commerce innovation",
    data: [
      { value: 178 },
      { value: 180 },
      { value: 182 },
      { value: 183 },
      { value: 184 },
      { value: 185 },
      { value: 185 },
    ],
  },
  {
    symbol: "META",
    name: "Meta Platforms Inc.",
    price: "$543.57",
    change: "-0.59%",
    isPositive: false,
    aiInsight: "Hold due to mixed signals in advertising and metaverse investments",
    data: [
      { value: 530 },
      { value: 535 },
      { value: 540 },
      { value: 545 },
      { value: 550 },
      { value: 545 },
      { value: 543 },
    ],
  },
  {
    symbol: "NFLX",
    name: "Netflix Inc.",
    price: "$918.29",
    change: "-0.37%",
    isPositive: false,
    aiInsight: "Hold as subscriber growth stabilizes amid increasing competition",
    data: [
      { value: 900 },
      { value: 905 },
      { value: 910 },
      { value: 915 },
      { value: 920 },
      { value: 925 },
      { value: 918 },
    ],
  },
  {
    symbol: "BABA",
    name: "Alibaba Group Holding Ltd.",
    price: "$107.73",
    change: "+3.45%",
    isPositive: true,
    aiInsight: "Buy on strong e-commerce recovery and cloud expansion",
    data: [
      { value: 100 },
      { value: 102 },
      { value: 104 },
      { value: 106 },
      { value: 108 },
      { value: 109 },
      { value: 107 },
    ],
  },
  {
    symbol: "INTC",
    name: "Intel Corp.",
    price: "$19.74",
    change: "-0.81%",
    isPositive: false,
    aiInsight: "Sell due to ongoing challenges in process node transitions",
    data: [
      { value: 20 },
      { value: 19.8 },
      { value: 19.6 },
      { value: 19.4 },
      { value: 19.2 },
      { value: 19 },
      { value: 19.7 },
    ],
  },
  {
    symbol: "AMD",
    name: "Advanced Micro Devices Inc.",
    price: "$93.40",
    change: "+5.19%",
    isPositive: true,
    aiInsight: "Strong buy on data center growth and AI chip leadership",
    data: [
      { value: 88 },
      { value: 89 },
      { value: 90 },
      { value: 91 },
      { value: 92 },
      { value: 93 },
      { value: 93 },
    ],
  },
  {
    symbol: "ORCL",
    name: "Oracle Corp.",
    price: "$132.35",
    change: "-0.79%",
    isPositive: false,
    aiInsight: "Hold as cloud transition continues amid competitive pressures",
    data: [
      { value: 130 },
      { value: 131 },
      { value: 132 },
      { value: 133 },
      { value: 134 },
      { value: 135 },
      { value: 132 },
    ],
  },
  {
    symbol: "CRM",
    name: "Salesforce Inc.",
    price: "$255.00",
    change: "-0.12%",
    isPositive: false,
    aiInsight: "Hold due to mixed enterprise demand and competitive landscape",
    data: [
      { value: 250 },
      { value: 252 },
      { value: 254 },
      { value: 256 },
      { value: 258 },
      { value: 257 },
      { value: 255 },
    ],
  },
] 

const TrendingStocks = () => {
  return (
    <Card className="data-card mt-6">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-xl flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Trending Stocks
        </CardTitle>
        <span className="text-sm text-muted-foreground">Recent Data</span>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {trendingStocks.map((stock) => (
            <div key={stock.symbol} className="flex items-center justify-between border-b border-border pb-4 last:border-0 last:pb-0">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <div className="font-bold">{stock.symbol}</div>
                  <div className="text-sm text-muted-foreground">{stock.name}</div>
                </div>
                <p className="mt-1 text-sm">{stock.aiInsight}</p>
              </div>
              <div className="flex items-end gap-4">
                <div className="text-right">
                  <div className="text-lg font-bold">{stock.price}</div>
                  <div className={cn(
                    "flex items-center justify-end text-sm",
                    stock.isPositive ? "text-quantai-green" : "text-quantai-red"
                  )}>
                    {stock.isPositive ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                    <span>{stock.change}</span>
                  </div>
                </div>
                <div className="h-12 w-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={stock.data}>
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke={stock.isPositive ? "#10B981" : "#EF4444"}
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default TrendingStocks;
