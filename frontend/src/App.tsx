import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import MarketInsights from "./pages/MarketInsights";
import Screener from "./pages/Screener";
import NotFound from "./pages/NotFound";
import Portfolio from "./pages/Portfolio";
import Compare from "./pages/Compare";
import Sectors from "./pages/Sectors";
import News from "./pages/News";
import Alerts from "./pages/Alerts";
import StockDetail from "./pages/StockDetail";

import MaazDashboard from './MaazDashboard';
import data from './data.json';
import UploadFileBox from './UploadFileBox';

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/market-insights" element={<MarketInsights />} />
          <Route path="/screener" element={<Screener />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/sectors" element={<Sectors />} />
          <Route path="/news" element={<News />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/stock/:symbol" element={<StockDetail />} />
          {/* <Route path="/stock/:symbol" element={<UploadFileBox />} /> */}

          {/* ✅ New Route for MaazDashboard */}
          <Route path="/dashboard" element={<MaazDashboard data={data} />} />

          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
