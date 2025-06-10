
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

// Mock data for AI insights
const aiInsights = [
  {
    title: "Tech Sector Outlook",
    summary: "AI-driven companies are likely to outperform the broader market in Q2. Cloud services showing resilience despite market volatility.",
    sentiment: "positive",
    date: "Apr 10, 2025",
  },
  {
    title: "Financial Sector Alert",
    summary: "Regional banks face headwinds from interest rate uncertainty. Larger institutions with diverse revenue streams remain better positioned.",
    sentiment: "neutral",
    date: "Apr 9, 2025",
  },
  {
    title: "Energy Market Shift",
    summary: "Traditional energy companies underperforming as policy shifts accelerate. Renewable energy investments showing momentum.",
    sentiment: "negative",
    date: "Apr 8, 2025",
  }
];

const AiInsights = () => {
  return (
    <Card className="data-card mt-6">
      <CardHeader>
        <CardTitle className="text-xl flex items-center gap-2">
          <Sparkles className="h-5 w-5" />
          QuantAI Insights of the Week
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {aiInsights.map((insight) => (
            <div key={insight.title} className="space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">{insight.title}</h3>
                <span className={cn(
                  "insight-tag",
                  insight.sentiment === "positive" && "insight-tag-positive",
                  insight.sentiment === "negative" && "insight-tag-negative",
                  insight.sentiment === "neutral" && "insight-tag-neutral"
                )}>
                  {insight.sentiment.charAt(0).toUpperCase() + insight.sentiment.slice(1)}
                </span>
              </div>
              <p className="text-muted-foreground">{insight.summary}</p>
              <div className="flex justify-between items-center pt-1">
                <span className="text-xs text-muted-foreground">{insight.date}</span>
                {/*<Button variant="ghost" size="sm" className="h-8 gap-1">
                  Read more <ArrowRight className="h-3 w-3" />
                </Button>*/}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default AiInsights;
