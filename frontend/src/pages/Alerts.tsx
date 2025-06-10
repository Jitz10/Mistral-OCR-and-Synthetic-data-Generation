import React, { useState } from "react";
import Layout from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bell, BellRing, Plus, Settings, ToggleLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";

const Alerts = () => {
  const [activeAlerts, setActiveAlerts] = useState(true);
  const [showAlertHistory, setShowAlertHistory] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [smsNotifications, setSmsNotifications] = useState(false);
  const [pushNotifications, setPushNotifications] = useState(true);

  const alerts = [
    { 
      type: "Price Alert", 
      status: "triggered",
      ticker: "NVDA", 
      condition: "Above $900", 
      triggerPrice: "$924.67",
      created: "Apr 10, 2025",
      triggered: "Apr 12, 2025"
    },
    { 
      type: "Technical Alert", 
      status: "active",
      ticker: "MSFT", 
      condition: "RSI crosses 70", 
      triggerPrice: "-",
      created: "Apr 8, 2025",
      triggered: "-"
    },
    { 
      type: "Earnings Alert", 
      status: "triggered",
      ticker: "AAPL", 
      condition: "Earnings announcement", 
      triggerPrice: "-",
      created: "Apr 5, 2025",
      triggered: "Apr 11, 2025"
    },
    { 
      type: "News Alert", 
      status: "active",
      ticker: "AMZN", 
      condition: "Major news", 
      triggerPrice: "-",
      created: "Apr 7, 2025",
      triggered: "-"
    },
    { 
      type: "Price Alert", 
      status: "triggered",
      ticker: "TSLA", 
      condition: "Below $200", 
      triggerPrice: "$197.42",
      created: "Apr 1, 2025",
      triggered: "Apr 9, 2025"
    },
  ];

  const alertHistoryData = [
    {
      date: "Apr 12, 2025",
      time: "10:24 AM",
      ticker: "NVDA",
      message: "NVIDIA crossed above $900 target price",
      notificationTypes: ["email", "push"]
    },
    {
      date: "Apr 11, 2025",
      time: "08:30 AM",
      ticker: "AAPL",
      message: "Apple earnings announcement scheduled for today after market close",
      notificationTypes: ["email", "push"]
    },
    {
      date: "Apr 9, 2025",
      time: "11:43 AM",
      ticker: "TSLA",
      message: "Tesla fell below $200 price target",
      notificationTypes: ["email", "push"]
    },
    {
      date: "Apr 7, 2025",
      time: "09:12 AM",
      ticker: "GOOGL",
      message: "Google announced new AI technology with potential revenue impact",
      notificationTypes: ["email"]
    },
    {
      date: "Apr 5, 2025",
      time: "02:30 PM",
      ticker: "META",
      message: "META RSI crossed above 70, now at 73.2",
      notificationTypes: ["push"]
    },
  ];

  const aiSuggestedAlerts = [
    {
      ticker: "JPM",
      type: "Technical",
      suggestion: "JPM forming bullish pattern, consider setting alert for breakout above $195",
      confidence: "high"
    },
    {
      ticker: "INTC",
      type: "Earnings",
      suggestion: "INTC earnings expected to beat estimates based on supplier data",
      confidence: "medium"
    },
    {
      ticker: "XOM",
      type: "Price",
      suggestion: "Consider setting support alert for XOM at $108 based on technical analysis",
      confidence: "medium"
    }
  ];

  return (
    <Layout>
      <div className="container space-y-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Alerts & Notifications</h1>
            <p className="text-muted-foreground">
              Manage your real-time market alerts and notification preferences
            </p>
          </div>
          <Button>
            <Plus className="mr-2 h-4 w-4" /> Create Alert
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Active Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center">
                <div className="text-2xl font-bold">{alerts.filter(a => a.status === "active").length}</div>
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bell className="h-6 w-6 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Triggered Alerts (Last 7 Days)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center">
                <div className="text-2xl font-bold">{alerts.filter(a => a.status === "triggered").length}</div>
                <div className="h-12 w-12 rounded-full bg-green-500/10 flex items-center justify-center">
                  <BellRing className="h-6 w-6 text-green-500" />
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Notification Settings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between items-center">
                <div className="text-2xl font-bold">3 Active</div>
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Settings className="h-6 w-6 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="active" className="w-full">
          <TabsList className="grid grid-cols-4 mb-6">
            <TabsTrigger value="active">Active Alerts</TabsTrigger>
            <TabsTrigger value="history">Alert History</TabsTrigger>
            <TabsTrigger value="suggested">AI Suggested</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="active">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Active Alerts</span>
                  <Button variant="outline" size="sm">Filter</Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left font-medium p-2 pl-0">Type</th>
                        <th className="text-left font-medium p-2">Ticker</th>
                        <th className="text-left font-medium p-2">Condition</th>
                        <th className="text-left font-medium p-2">Status</th>
                        <th className="text-left font-medium p-2">Created</th>
                        <th className="text-left font-medium p-2">Triggered</th>
                        <th className="text-right font-medium p-2 pr-0">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {alerts.map((alert, index) => (
                        <tr key={index} className="border-b">
                          <td className="p-2 pl-0">
                            <Badge variant="outline" className="bg-primary/5">
                              {alert.type}
                            </Badge>
                          </td>
                          <td className="p-2 font-medium">{alert.ticker}</td>
                          <td className="p-2">{alert.condition}</td>
                          <td className="p-2">
                            <Badge variant={alert.status === "active" ? "secondary" : "default"}>
                              {alert.status === "active" ? "Active" : "Triggered"}
                            </Badge>
                          </td>
                          <td className="p-2">{alert.created}</td>
                          <td className="p-2">{alert.triggered}</td>
                          <td className="text-right p-2 pr-0">
                            <Button variant="ghost" size="sm">Edit</Button>
                            <Button variant="ghost" size="sm">Delete</Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Alert History</span>
                  <Button variant="outline" size="sm">Export</Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {alertHistoryData.map((alert, index) => (
                    <div key={index} className="flex items-start gap-4 p-4 border rounded-lg">
                      <div className="rounded-full bg-primary/10 p-2 mt-0.5">
                        <BellRing className="h-4 w-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <h4 className="font-medium">{alert.ticker}</h4>
                          <div className="text-xs text-muted-foreground">
                            {alert.date} • {alert.time}
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{alert.message}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className="text-xs text-muted-foreground">Delivered via:</span>
                          <div className="flex gap-1">
                            {alert.notificationTypes.includes("email") && (
                              <Badge variant="outline" className="text-xs">Email</Badge>
                            )}
                            {alert.notificationTypes.includes("sms") && (
                              <Badge variant="outline" className="text-xs">SMS</Badge>
                            )}
                            {alert.notificationTypes.includes("push") && (
                              <Badge variant="outline" className="text-xs">Push</Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="suggested">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ToggleLeft className="h-5 w-5" />
                  AI-Suggested Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-6">
                  Based on your portfolio, watchlist, and market conditions, QuantAI suggests the following alerts:
                </p>
                <div className="space-y-4">
                  {aiSuggestedAlerts.map((alert, index) => (
                    <div key={index} className="flex items-start justify-between gap-4 p-4 border rounded-lg">
                      <div className="flex gap-4">
                        <div className="rounded-full bg-primary/10 p-2 mt-0.5">
                          <Bell className="h-4 w-4 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium">{alert.ticker}</h4>
                            <Badge variant="outline">{alert.type}</Badge>
                            <Badge variant={
                              alert.confidence === "high" ? "default" : 
                              alert.confidence === "medium" ? "secondary" : "outline"
                            }>
                              {alert.confidence} confidence
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">{alert.suggestion}</p>
                        </div>
                      </div>
                      <Button size="sm">Create</Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="settings">
            <Card>
              <CardHeader>
                <CardTitle>Notification Preferences</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="space-y-4">
                    <h3 className="font-medium">Alert Types</h3>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">Price Alerts</Label>
                        <p className="text-sm text-muted-foreground">
                          Notifications when stocks hit your price targets
                        </p>
                      </div>
                      <Switch checked={activeAlerts} onCheckedChange={setActiveAlerts} />
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">Technical Indicators</Label>
                        <p className="text-sm text-muted-foreground">
                          Alerts for RSI, MACD, and other technical signals
                        </p>
                      </div>
                      <Switch checked={showAlertHistory} onCheckedChange={setShowAlertHistory} />
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">News & Events</Label>
                        <p className="text-sm text-muted-foreground">
                          Major news and corporate events affecting your watchlist
                        </p>
                      </div>
                      <Switch checked={true} />
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">AI-Generated Insights</Label>
                        <p className="text-sm text-muted-foreground">
                          Advanced pattern recognition and predictive alerts
                        </p>
                      </div>
                      <Switch checked={true} />
                    </div>
                  </div>

                  <div className="space-y-4 pt-4 border-t">
                    <h3 className="font-medium">Delivery Methods</h3>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">Email Notifications</Label>
                        <p className="text-sm text-muted-foreground">
                          Receive alerts via email
                        </p>
                      </div>
                      <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">SMS Notifications</Label>
                        <p className="text-sm text-muted-foreground">
                          Receive alerts via text message
                        </p>
                      </div>
                      <Switch checked={smsNotifications} onCheckedChange={setSmsNotifications} />
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <div className="space-y-0.5">
                        <Label className="text-base">Push Notifications</Label>
                        <p className="text-sm text-muted-foreground">
                          Receive alerts via browser or mobile app notifications
                        </p>
                      </div>
                      <Switch checked={pushNotifications} onCheckedChange={setPushNotifications} />
                    </div>
                  </div>

                  <div className="space-y-4 pt-4 border-t">
                    <h3 className="font-medium">Notification Timing</h3>
                    <div className="space-y-2">
                      <Label>Delivery Priority</Label>
                      <RadioGroup defaultValue="realtime">
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="realtime" id="realtime" />
                          <Label htmlFor="realtime">Real-time (immediate delivery)</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="batched" id="batched" />
                          <Label htmlFor="batched">Batched (delivered hourly)</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="daily" id="daily" />
                          <Label htmlFor="daily">Daily digest (once per day)</Label>
                        </div>
                      </RadioGroup>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
};

export default Alerts;
