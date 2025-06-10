interface ESGCompany {
  rank: number;
  company: string;
  ticker: string;
  score: number;
  environment: string;
  social: string;
  governance: string;
}

interface SectorOutlook {
  overview: string;
  keyTrends: string[];
  capitalAllocation: string;
}

interface SectorData {
  esgRankings: Record<string, ESGCompany[]>;
  aiOutlook: Record<string, SectorOutlook>;
}

export const sectorData: SectorData = {
  esgRankings: {
    "basic-materials": [
      { rank: 1, company: "Linde", ticker: "LIN", score: 90, environment: "A", social: "B+", governance: "A" },
      { rank: 2, company: "Umicore", ticker: "UMI", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 3, company: "Chr. Hansen Holding A/S", ticker: "CHR", score: 85, environment: "B", social: "B", governance: "B+" },
      { rank: 4, company: "BASF", ticker: "BAS", score: 82, environment: "B", social: "B", governance: "B" },
      { rank: 5, company: "Air Liquide", ticker: "AI", score: 80, environment: "B", social: "B", governance: "C" }
    ],
    "communication-services": [
      { rank: 1, company: "RELX", ticker: "RELX", score: 93, environment: "A+", social: "A", governance: "A" },
      { rank: 2, company: "Verizon Communications", ticker: "VZ", score: 89, environment: "A", social: "B+", governance: "B+" },
      { rank: 3, company: "AT&T", ticker: "T", score: 86, environment: "B+", social: "B+", governance: "B+" },
      { rank: 4, company: "Comcast", ticker: "CMCSA", score: 84, environment: "B", social: "B+", governance: "B" },
      { rank: 5, company: "T-Mobile US", ticker: "TMUS", score: 82, environment: "B", social: "B", governance: "C" }
    ],
    "consumer-cyclical": [
      { rank: 1, company: "Best Buy", ticker: "BBY", score: 91, environment: "A", social: "A", governance: "A" },
      { rank: 2, company: "AutoZone", ticker: "AZO", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 3, company: "O'Reilly Automotive", ticker: "ORLY", score: 86, environment: "B+", social: "B+", governance: "B+" },
      { rank: 4, company: "Lowe's Companies", ticker: "LOW", score: 84, environment: "B", social: "B+", governance: "B" },
      { rank: 5, company: "Home Depot", ticker: "HD", score: 82, environment: "B", social: "B", governance: "B" }
    ],
    "consumer-defensive": [
      { rank: 1, company: "The J. M. Smucker Company", ticker: "SJM", score: 90, environment: "A", social: "B+", governance: "A" },
      { rank: 2, company: "Walgreens Boots Alliance", ticker: "WBA", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 3, company: "Hormel Foods Corporation", ticker: "HRL", score: 87, environment: "B+", social: "A", governance: "B+" },
      { rank: 4, company: "The Estée Lauder Companies Inc", ticker: "EL", score: 85, environment: "B", social: "B+", governance: "B" },
      { rank: 5, company: "Sysco Corporation", ticker: "SYY", score: 83, environment: "B", social: "B", governance: "C" }
    ],
    "energy": [
      { rank: 1, company: "Ørsted", ticker: "ORSTED", score: 92, environment: "A+", social: "A", governance: "A" },
      { rank: 2, company: "Vestas Wind Systems", ticker: "VWS", score: 90, environment: "A+", social: "A", governance: "B+" },
      { rank: 3, company: "Enel", ticker: "ENEL", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 4, company: "NextEra Energy", ticker: "NEE", score: 86, environment: "B+", social: "B+", governance: "B+" },
      { rank: 5, company: "Iberdrola", ticker: "IBE", score: 85, environment: "B", social: "B", governance: "B" }
    ],
    "financial-services": [
      { rank: 1, company: "S&P Global", ticker: "SPGI", score: 89, environment: "A", social: "B+", governance: "A" },
      { rank: 2, company: "Mastercard", ticker: "MA", score: 87, environment: "B+", social: "B+", governance: "A" },
      { rank: 3, company: "Visa", ticker: "V", score: 85, environment: "B", social: "B+", governance: "B" },
      { rank: 4, company: "PayPal", ticker: "PYPL", score: 83, environment: "B", social: "B", governance: "C" },
      { rank: 5, company: "American Express", ticker: "AXP", score: 81, environment: "C", social: "C", governance: "B" }
    ],
    "healthcare": [
      { rank: 1, company: "Idexx Laboratories", ticker: "IDXX", score: 91, environment: "A", social: "A", governance: "A" },
      { rank: 2, company: "Pfizer", ticker: "PFE", score: 89, environment: "A", social: "A", governance: "B+" },
      { rank: 3, company: "Johnson & Johnson", ticker: "JNJ", score: 87, environment: "B+", social: "B+", governance: "A" },
      { rank: 4, company: "Merck & Co.", ticker: "MRK", score: 85, environment: "B", social: "B", governance: "B+" },
      { rank: 5, company: "AbbVie", ticker: "ABBV", score: 83, environment: "B", social: "B", governance: "C" }
    ],
    "industrials": [
      { rank: 1, company: "Keysight Technologies", ticker: "KEYS", score: 94, environment: "A+", social: "A", governance: "A+" },
      { rank: 2, company: "Woodward", ticker: "WWD", score: 91, environment: "A", social: "A", governance: "A" },
      { rank: 3, company: "Kone Oyj", ticker: "KNEBV", score: 89, environment: "A", social: "B+", governance: "B+" },
      { rank: 4, company: "Siemens", ticker: "SIEGY", score: 87, environment: "B+", social: "B+", governance: "A" },
      { rank: 5, company: "Schneider Electric", ticker: "SBGSF", score: 85, environment: "B", social: "B+", governance: "B" }
    ],
    "real-estate": [
      { rank: 1, company: "Prologis", ticker: "PLD", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 2, company: "Equinix", ticker: "EQIX", score: 86, environment: "B+", social: "B+", governance: "B+" },
      { rank: 3, company: "Digital Realty", ticker: "DLR", score: 84, environment: "B", social: "B", governance: "B" },
      { rank: 4, company: "Welltower", ticker: "WELL", score: 82, environment: "B", social: "B", governance: "C" },
      { rank: 5, company: "Public Storage", ticker: "PSA", score: 80, environment: "C", social: "C", governance: "B" }
    ],
    "technology": [
      { rank: 1, company: "Microsoft", ticker: "MSFT", score: 95, environment: "A+", social: "A+", governance: "A+" },
      { rank: 2, company: "Nvidia", ticker: "NVDA", score: 93, environment: "A+", social: "A", governance: "A" },
      { rank: 3, company: "Adobe", ticker: "ADBE", score: 91, environment: "A", social: "A", governance: "A" },
      { rank: 4, company: "Salesforce", ticker: "CRM", score: 89, environment: "A", social: "B+", governance: "B+" },
      { rank: 5, company: "Intuit", ticker: "INTU", score: 87, environment: "B+", social: "B+", governance: "A" }
    ],
    "utilities": [
      { rank: 1, company: "Southern Company", ticker: "SO", score: 90, environment: "A", social: "B+", governance: "A" },
      { rank: 2, company: "Alliant Energy", ticker: "LNT", score: 88, environment: "B+", social: "B+", governance: "A" },
      { rank: 3, company: "NRG Energy", ticker: "NRG", score: 86, environment: "B+", social: "B+", governance: "B+" },
      { rank: 4, company: "Dominion Energy", ticker: "D", score: 84, environment: "B", social: "B", governance: "B" },
      { rank: 5, company: "Duke Energy", ticker: "DUK", score: 82, environment: "B", social: "B", governance: "C" }
    ]
  },
  aiOutlook: {
    "basic-materials": {
      overview: "The Basic Materials sector is experiencing moderate growth, supported by infrastructure development and increased demand for construction materials. However, global trade tensions, particularly U.S. tariffs, are impacting export-driven segments like metals and chemicals. The sector's average P/E ratio stands at 17.5x, slightly below the broader market average.",
      keyTrends: [
        "Metals & Mining: Domestic demand remains strong, but exports face headwinds due to international trade policies.",
        "Chemicals: Specialty chemicals are witnessing robust demand, while basic chemicals are affected by fluctuating raw material prices.",
        "Cement: Urbanization and infrastructure projects are driving steady growth, with companies reporting improved margins."
      ],
      capitalAllocation: "Capital allocation is focused on expanding production capacities and investing in sustainable practices to meet environmental regulations."
    },
    "communication-services": {
      overview: "The Communication Services sector continues to expand, driven by increased digital consumption and the rollout of 5G services. The sector's average P/E ratio is approximately 22.3x, indicating healthy investor confidence.",
      keyTrends: [
        "Telecom: 5G deployment is accelerating, enhancing connectivity and opening new revenue streams.",
        "Media & Entertainment: Digital platforms are experiencing significant user growth, with content consumption at an all-time high.",
        "Internet Services: E-commerce and online services are expanding rapidly, supported by increased internet penetration."
      ],
      capitalAllocation: "Investments are being channeled into network infrastructure and content creation to capture the growing digital audience."
    },
    "consumer-cyclical": {
      overview: "The Consumer Cyclical sector is showing signs of recovery, with an average P/E ratio of 24.1x, reflecting optimistic consumer sentiment. Rural consumption is expected to improve due to easing food inflation and favorable monsoon forecasts. However, global trade tensions may affect export-oriented consumer goods.",
      keyTrends: [
        "Automobiles: Sales are rebounding, particularly in the electric vehicle segment, supported by government incentives.",
        "Retail: Brick-and-mortar stores are recovering, while online retail continues to grow, driven by changing consumer behaviors.",
        "Hospitality: Tourism is picking up, leading to increased occupancy rates and revenue growth in the hospitality industry."
      ],
      capitalAllocation: "Companies are investing in digital transformation and supply chain optimization to enhance customer experience and operational efficiency."
    },
    "consumer-defensive": {
      overview: "The Consumer Defensive sector remains stable, with consistent demand for essential goods. The sector's average P/E ratio is around 20.8x, indicating steady investor interest.",
      keyTrends: [
        "Food & Beverages: Companies are focusing on health-oriented products, catering to changing consumer preferences.",
        "Household Products: Demand remains strong, with increased focus on hygiene and sanitation products.",
        "Personal Care: Growth is driven by premiumization and expansion into rural markets."
      ],
      capitalAllocation: "Investments are directed towards expanding distribution networks and product innovation to maintain market share."
    },
    "energy": {
      overview: "The Energy sector is undergoing a transformation, with a shift towards renewable sources. The sector's average P/E ratio stands at 15.2x, reflecting cautious optimism.",
      keyTrends: [
        "Oil & Gas: Traditional energy sectors are adapting to the transition, focusing on cleaner technologies.",
        "Renewables: India is advancing towards its renewable energy targets, with significant investments in solar and wind projects.",
        "Power Utilities: Efforts are underway to enhance electricity access and reliability across rural regions."
      ],
      capitalAllocation: "Capital is being allocated to modernize infrastructure and invest in sustainable energy solutions."
    },
    "financial-services": {
      overview: "The Financial Services sector is experiencing robust growth, with an average P/E ratio of 18.7x. The Reserve Bank of India's accommodative stance, including recent rate cuts, is expected to stimulate credit growth.",
      keyTrends: [
        "Banking: Credit demand is rising, supported by lower interest rates and improved asset quality.",
        "Insurance: The sector is expanding, driven by increased awareness and regulatory support.",
        "Fintech: Digital banking and fintech innovations are enhancing financial inclusion and service delivery."
      ],
      capitalAllocation: "Investments are focused on digital infrastructure and expanding financial services to underserved regions."
    },
    "healthcare": {
      overview: "The Healthcare sector is expanding rapidly, with an average P/E ratio of 25.4x. The sector is driven by increased healthcare awareness and government initiatives like Ayushman Bharat.",
      keyTrends: [
        "Pharmaceuticals: Pharmaceutical exports are growing, with India capitalizing on global demand for generic medicines.",
        "Hospitals: Private healthcare providers are expanding, catering to rising demand for quality healthcare services.",
        "Medical Devices: The sector is witnessing growth, supported by government incentives and increased domestic manufacturing."
      ],
      capitalAllocation: "Capital allocation is directed towards research and development, as well as expanding healthcare infrastructure."
    },
    "industrials": {
      overview: "The Industrials sector is benefiting from government initiatives such as the National Infrastructure Pipeline. The sector's average P/E ratio is approximately 19.9x, indicating positive investor sentiment driven by increased infrastructure spending and the 'Make in India' push.",
      keyTrends: [
        "Construction & Engineering: Public infrastructure projects including highways, metros, and smart cities are fueling order books and revenue growth.",
        "Capital Goods: Demand is rising from both domestic and export markets, especially in automation, robotics, and green machinery.",
        "Logistics: Strong e-commerce growth and supply chain digitization are enhancing efficiency and margins."
      ],
      capitalAllocation: "Capital is flowing into capacity expansion, supply chain modernization, and adoption of Industry 4.0 technologies to boost productivity and export competitiveness."
    },
    "real-estate": {
      overview: "The Real Estate sector is seeing a strong recovery, with the residential market leading the rebound. The average P/E ratio for listed developers is 21.3x, up 18% YoY, reflecting growing buyer confidence and low interest rate support.",
      keyTrends: [
        "Residential: Demand for mid- to premium-housing remains robust, particularly in urban centers, supported by steady pricing and flexible financing.",
        "Commercial: Office space absorption is improving with return-to-office trends, especially in IT/ITES and co-working spaces.",
        "Retail Real Estate: Retail leasing is picking up as footfalls return to pre-pandemic levels across major malls."
      ],
      capitalAllocation: "Developers are focusing on deleveraging, launching new projects in top-tier cities, and incorporating green building standards to attract ESG-aligned investors."
    },
    "technology": {
      overview: "The Technology sector continues to demonstrate resilience despite valuation pressures, with an average P/E ratio of 32.4x representing a 45% premium to the broader market. AI remains the primary growth catalyst, with companies reporting AI-related revenue growth averaging 85% YoY in Q1 2025.",
      keyTrends: [
        "Semiconductors: Supply constraints easing while demand remains robust, particularly in data center and AI acceleration chips where margins continue to expand.",
        "Software: Enterprise spending resilience with cloud workloads growing 29% YoY, though contract scrutiny and sales cycles are lengthening.",
        "Hardware: Mixed outlook with consumer segments facing cyclical pressure while enterprise demand shows early signs of upgrade cycle driven by AI capabilities."
      ],
      capitalAllocation: "Capital allocation strategy across the sector is evolving, with R&D intensity increasing 320bps YoY focused predominantly on AI capabilities, while shareholder returns (dividends + buybacks) have moderated to 67% of free cash flow from 78% a year ago."
    },
    "utilities": {
      overview: "The Utilities sector is undergoing a structural shift toward green and digital infrastructure. With a modest P/E ratio of 13.2x, the sector offers a defensive play with improving fundamentals, especially in renewable integration.",
      keyTrends: [
        "Electric Utilities: Modernization and smart grid projects are boosting operational efficiency and reducing losses.",
        "Renewable Power: Wind and solar capacity additions are ahead of target, with hybrid storage solutions gaining traction.",
        "Water & Waste Management: Urban infrastructure projects are expanding, supported by Swachh Bharat and AMRUT initiatives."
      ],
      capitalAllocation: "Investments are accelerating in sustainability, digital metering, and energy storage solutions, making the sector more competitive and future-ready."
    }
  }
};

// Function to get ESG rankings for a sector
export const getESGRankings = (sector: string): ESGCompany[] => {
  return sectorData.esgRankings[sector] || [];
};

// Function to get AI outlook for a sector
export const getAISectorOutlook = (sector: string): SectorOutlook | null => {
  return sectorData.aiOutlook[sector] || null;
};

// Function to generate dynamic outlook for other sectors
export const generateDynamicOutlook = (sector: string, metrics: {
  ytdReturn: number;
  peRatio: number;
  dividendYield: number;
  marketCap: number;
}): SectorOutlook => {
  const { ytdReturn, peRatio, dividendYield, marketCap } = metrics;

  return {
    overview: `The ${sector} sector exhibits ${ytdReturn > 0 ? 'positive' : 'negative'} performance year-to-date (${ytdReturn}%), influenced by ${ytdReturn > 10 ? 'strong growth catalysts and favorable market conditions' : ytdReturn > 0 ? 'moderate growth and stable market conditions' : 'significant headwinds and challenging market dynamics'}.`,
    keyTrends: [
      `Average P/E ratio of ${peRatio}x suggests ${peRatio > 25 ? 'premium valuations reflecting growth expectations' : peRatio > 15 ? 'moderate valuations aligned with market averages' : 'value-oriented positioning reflecting growth concerns'}`,
      `Dividend yield of ${dividendYield}% indicates ${dividendYield > 3 ? 'strong income characteristics' : dividendYield > 1.5 ? 'balanced approach to shareholder returns' : 'focus on reinvestment and growth'}`,
      `Market concentration remains ${marketCap > 10000 ? 'heavily weighted toward mega-cap leaders' : marketCap > 5000 ? 'dominated by established large-cap companies' : 'relatively balanced across market cap segments'}`
    ],
    capitalAllocation: `Current sentiment indicates ${ytdReturn > 15 ? 'strong bullish momentum' : ytdReturn > 5 ? 'cautious optimism' : ytdReturn > 0 ? 'neutral positioning' : 'defensive positioning'} with institutional investors ${ytdReturn > 10 ? 'overweight' : ytdReturn > 0 ? 'market weight' : 'underweight'} relative to benchmark allocations.`
  };
}; 