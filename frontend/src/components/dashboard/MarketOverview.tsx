
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUp, ArrowDown, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

type IndexData = {
  name: string;
  value: string;
  change: number;
  percentChange: number;
};

const marketIndices: IndexData[] = [
    {
      "name": "S&P 500",
      "value": "5,339.40",
      "change": 9.23,
      "percentChange": 0.17
    },
    {
      "name": "Nasdaq",
      "value": "16,552.90",
      "change": 7.90,
      "percentChange": 0.18
    },
    {
      "name": "Dow Jones",
      "value": "40,191.00",
      "change": 6.26,
      "percentChange": 0.16
    },
    {
      "name": "Russell 2000",
      "value": "1,843.60",
      "change": 2.74,
      "percentChange": 0.15
    }
  
];

export function MarketOverview() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {marketIndices.map((index) => (
        <Card key={index.name} className="overflow-hidden">
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-sm font-medium flex items-center justify-between">
              <span>{index.name}</span>
              <Badge
                variant="outline"
                className={cn(
                  "ml-2",
                  index.percentChange >= 0 ? "text-quantai-green" : "text-quantai-red"
                )}
              >
                <Activity size={12} className="mr-1" />
                Most Recent
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1">
            <div className="text-2xl font-bold">{index.value}</div>
            <div className="flex items-center mt-1">
              <div
                className={cn(
                  "flex items-center text-sm",
                  index.percentChange >= 0 ? "text-quantai-green" : "text-quantai-red"
                )}
              >
                {index.percentChange >= 0 ? (
                  <ArrowUp size={16} className="mr-1" />
                ) : (
                  <ArrowDown size={16} className="mr-1" />
                )}
                <span>{Math.abs(index.change).toFixed(2)}</span>
                <span className="ml-1">
                  ({Math.abs(index.percentChange).toFixed(2)}%)
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}