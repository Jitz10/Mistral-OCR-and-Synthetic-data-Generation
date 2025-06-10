export interface PortfolioHolding {
    ticker: string;
    name: string;
    shares: number;
    avgCost: number;
    currentPrice: number;
    marketValue: number;
    pl: number;
    plPercent: number;
    aiRec: string;
    sector: string;
    sparklineData: number[];
  }
  
  export const initialPortfolioHoldings: PortfolioHolding[] = [
    { ticker: 'AAPL', name: 'Apple Inc.', shares: 25, avgCost: 155.42, currentPrice: 187.68, marketValue: 4692, pl: 805.5, plPercent: 20.7, aiRec: 'Buy', sector: 'Technology', sparklineData: Array.from({length: 20}, () => Math.random() * 10 + 180) },
    { ticker: 'MSFT', name: 'Microsoft Corp.', shares: 12, avgCost: 342.15, currentPrice: 405.12, marketValue: 4861.44, pl: 755.64, plPercent: 18.4, aiRec: 'Buy', sector: 'Technology', sparklineData: Array.from({length: 20}, () => Math.random() * 15 + 390) },
    { ticker: 'AMZN', name: 'Amazon.com Inc.', shares: 18, avgCost: 145.22, currentPrice: 173.45, marketValue: 3122.1, pl: 508.14, plPercent: 19.4, aiRec: 'Buy', sector: 'Consumer Cyclical', sparklineData: Array.from({length: 20}, () => Math.random() * 8 + 165) },
    { ticker: 'GOOGL', name: 'Alphabet Inc.', shares: 20, avgCost: 128.75, currentPrice: 151.35, marketValue: 3027, pl: 452, plPercent: 17.6, aiRec: 'Buy', sector: 'Communication Services', sparklineData: Array.from({length: 20}, () => Math.random() * 6 + 145) },
    { ticker: 'TSLA', name: 'Tesla, Inc.', shares: 15, avgCost: 312.45, currentPrice: 273.54, marketValue: 4103.1, pl: -584.65, plPercent: -12.5, aiRec: 'Hold', sector: 'Consumer Cyclical', sparklineData: Array.from({length: 20}, () => Math.random() * 12 + 265) },
    { ticker: 'JPM', name: 'JPMorgan Chase & Co.', shares: 10, avgCost: 172.34, currentPrice: 197.45, marketValue: 1974.5, pl: 251.1, plPercent: 14.6, aiRec: 'Buy', sector: 'Financial Services', sparklineData: Array.from({length: 20}, () => Math.random() * 7 + 190) },
    { ticker: 'JNJ', name: 'Johnson & Johnson', shares: 12, avgCost: 165.22, currentPrice: 154.78, marketValue: 1857.36, pl: -124.32, plPercent: -6.3, aiRec: 'Hold', sector: 'Healthcare', sparklineData: Array.from({length: 20}, () => Math.random() * 4 + 152) },
    { ticker: 'V', name: 'Visa Inc.', shares: 8, avgCost: 222.45, currentPrice: 248.32, marketValue: 1986.56, pl: 206.96, plPercent: 11.6, aiRec: 'Buy', sector: 'Financial Services', sparklineData: Array.from({length: 20}, () => Math.random() * 8 + 240) },
  ];
  
  export const sectors = [
    'Technology',
    'Consumer Cyclical',
    'Financial Services',
    'Healthcare',
    'Communication Services',
    'Energy',
    'Materials',
    'Utilities',
    'Real Estate',
    'Industrial',
    'Consumer Defensive'
  ];