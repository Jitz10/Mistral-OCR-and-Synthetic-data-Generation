import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Grid3X3 } from "lucide-react";
import { useQueries } from '@tanstack/react-query';

interface SectorPerformance {
  name: string;
  performance: number;
  isPositive: boolean;
}

const sectors = [
  "basic-materials",
  "communication-services",
  "consumer-cyclical",
  "consumer-defensive",
  "energy",
  "financial-services",
  "healthcare",
  "industrials",
  "real-estate",
  "technology",
  "utilities",
];

const displayNames: Record<string, string> = {
  "basic-materials": "Materials",
  "communication-services": "Communication",
  "consumer-cyclical": "Consumer Disc.",
  "consumer-defensive": "Consumer Staples",
  "energy": "Energy",
  "financial-services": "Financials",
  "healthcare": "Healthcare",
  "industrials": "Industrials",
  "real-estate": "Real Estate",
  "technology": "Technology",
  "utilities": "Utilities",
};

const getHeatmapColor = (performance: number, isPositive: boolean) => {
  const absPerformance = Math.abs(performance);
  
  if (isPositive) {
    if (absPerformance > 2) return "bg-green-600 text-white";
    if (absPerformance > 1) return "bg-green-500 text-white";
    if (absPerformance > 0.5) return "bg-green-400 text-white";
    return "bg-green-300 text-green-950";
  } else {
    if (absPerformance > 2) return "bg-red-600 text-white";
    if (absPerformance > 1) return "bg-red-500 text-white";
    if (absPerformance > 0.5) return "bg-red-400 text-white";
    return "bg-red-300 text-red-950";
  }
};

const SectorHeatmap = () => {
  // Fetch data for all sectors using useQueries
  const sectorQueries = useQueries({
    queries: sectors.map(sector => ({
      queryKey: ['sectorInfo', sector],
      queryFn: async () => {
        const response = await fetch('http://127.0.0.1:5000/sector_info', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sector }),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch data for ${sector}`);
        }

        const data = await response.json();
        return {
          name: sector,
          performance: data.fiftyDayAverageChangePercent * 100, // Convert to percentage
          isPositive: data.fiftyDayAverageChangePercent >= 0,
        } as SectorPerformance;
      },
      retry: 1,
      staleTime: 1000 * 60 * 5, // Cache for 5 minutes
    })),
  });

  // Process query results
  const sectorPerformance: SectorPerformance[] = sectorQueries.map((query, index) => {
    if (query.isLoading || query.isError) {
      return {
        name: sectors[index],
        performance: 0,
        isPositive: true,
      };
    }
    return query.data!;
  });

  return (
    <Card className="data-card mt-6">
      <CardHeader>
        <CardTitle className="text-xl flex items-center gap-2">
          <Grid3X3 className="h-5 w-5" />
          Sector Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        {sectorQueries.some(query => query.isLoading) ? (
          <div className="text-center">Loading...</div>
        ) : sectorQueries.some(query => query.isError) ? (
          <div className="text-center text-red-500">Error loading some sector data</div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {sectorPerformance.map((sector) => (
              <div
                key={sector.name}
                className={cn(
                  "rounded-md p-3 transition-all hover:opacity-90",
                  getHeatmapColor(sector.performance, sector.isPositive)
                )}
              >
                <div className="font-medium truncate">{displayNames[sector.name]}</div>
                <div className="text-sm font-bold">
                  {sector.isPositive ? "+" : ""}
                  {sector.performance.toFixed(2)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SectorHeatmap;