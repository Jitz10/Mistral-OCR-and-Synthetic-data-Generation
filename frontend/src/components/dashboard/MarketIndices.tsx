
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

// Mock data for market indices
const marketIndices = [
  {
    name: "S&P 500",
    value: "5,123.78",
    change: "+1.24%",
    isPositive: true,
    data: [
      { value: 5000 },
      { value: 5050 },
      { value: 5010 },
      { value: 5080 },
      { value: 5070 },
      { value: 5100 },
      { value: 5123 },
    ]
  },
  {
    name: "NASDAQ",
    value: "16,492.16",
    change: "+1.56%",
    isPositive: true,
    data: [
      { value: 16100 },
      { value: 16250 },
      { value: 16180 },
      { value: 16300 },
      { value: 16380 },
      { value: 16450 },
      { value: 16492 },
    ]
  },
  {
    name: "Dow Jones",
    value: "39,103.40",
    change: "+0.87%",
    isPositive: true,
    data: [
      { value: 38800 },
      { value: 38900 },
      { value: 38850 },
      { value: 38950 },
      { value: 39000 },
      { value: 39050 },
      { value: 39103 },
    ]
  },
  {
    name: "Russell 2000",
    value: "2,042.60",
    change: "-0.36%",
    isPositive: false,
    data: [
      { value: 2070 },
      { value: 2060 },
      { value: 2080 },
      { value: 2050 },
      { value: 2045 },
      { value: 2048 },
      { value: 2043 },
    ]
  }
];

const MarketIndices = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
      {marketIndices.map((index) => (
        <Card key={index.name} className="data-card-hover">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium">{index.name}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-end justify-between">
              <div>
                <p className="text-lg font-bold">{index.value}</p>
                <p className={cn(
                  "flex items-center text-sm",
                  index.isPositive ? "text-quantai-green" : "text-quantai-red"
                )}>
                  {index.isPositive ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                  <span>{index.change}</span>
                </p>
              </div>
              <div className="h-12 w-28">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={index.data}>
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke={index.isPositive ? "#10B981" : "#EF4444"} 
                      strokeWidth={2} 
                      dot={false} 
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default MarketIndices;
