export interface SwotData {
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
}

export interface SectorSwotData {
  [key: string]: SwotData;
}

export const sectorSwotData: SectorSwotData = {
  "basic-materials": {
    "strengths": [
      "Essential for manufacturing and construction industries",
      "Reliable demand from developing economies",
      "High barriers to entry for competitors",
      "Typically have strong cash flows during economic growth"
    ],
    "weaknesses": [
      "Vulnerable to commodity price fluctuations",
      "High capital expenditure requirements",
      "Environmental regulation compliance costs",
      "Cyclical performance tied to economic conditions"
    ],
    "opportunities": [
      "Growing demand from green technology and infrastructure projects",
      "Expansion into emerging markets",
      "Innovation in sustainable materials and processes",
      "Strategic mergers and acquisitions for vertical integration"
    ],
    "threats": [
      "Increasing environmental regulations",
      "Supply chain disruptions",
      "Competition from recycled materials",
      "Political instability in resource-rich regions"
    ]
  },
  "communication-services": {
    "strengths": [
      "Essential service with consistent consumer demand",
      "High switching costs for established customers",
      "Recurring revenue streams through subscriptions",
      "Established infrastructure networks provide competitive advantage"
    ],
    "weaknesses": [
      "High infrastructure maintenance costs",
      "Regulatory hurdles across different markets",
      "Rapid technological obsolescence",
      "Price competition eroding profit margins"
    ],
    "opportunities": [
      "5G network expansion and monetization",
      "Growth in streaming content and digital media",
      "IoT and smart device connectivity solutions",
      "Expansion into underserved markets"
    ],
    "threats": [
      "Intense competition from tech companies",
      "Regulatory changes affecting pricing and operations",
      "Cybersecurity vulnerabilities and data breaches",
      "Consumer privacy concerns"
    ]
  },
  "consumer-cyclical": {
    "strengths": [
      "Brand loyalty drives repeat business",
      "Adaptable to changing consumer preferences",
      "Multiple revenue channels including e-commerce",
      "Product diversification reduces risk"
    ],
    "weaknesses": [
      "Highly sensitive to economic downturns",
      "Seasonal revenue fluctuations",
      "Reliance on discretionary consumer spending",
      "Changing fashion trends create inventory risks"
    ],
    "opportunities": [
      "Growing middle class in emerging markets",
      "E-commerce expansion and direct-to-consum consumer models",
      "Personalization through data analytics",
      "Sustainable product offerings appealing to eco-conscious consumers"
    ],
    "threats": [
      "Recession risk impacts discretionary spending",
      "Fast-changing consumer preferences",
      "Supply chain vulnerabilities",
      "Intense competition from online retailers"
    ]
  },
  "consumer-defensive": {
    "strengths": [
      "Stable demand regardless of economic conditions",
      "Strong cash flow generation",
      "Brand loyalty enables pricing power",
      "Low business model disruption risk"
    ],
    "weaknesses": [
      "Limited growth potential in mature markets",
      "Thin profit margins requiring volume sales",
      "Rising commodity input costs",
      "Limited pricing power in competitive segments"
    ],
    "opportunities": [
      "Health-conscious product innovations",
      "Expansion into emerging markets",
      "Private label offerings for cost-conscious consumers",
      "Direct-to-consumer distribution channels"
    ],
    "threats": [
      "Changing consumer preferences toward healthier alternatives",
      "Rising competition from store brands",
      "Regulatory changes affecting ingredients or packaging",
      "Inflation pressures on production costs"
    ]
  },
  "energy": {
    "strengths": [
      "Essential for global economic function",
      "High barriers to entry",
      "Established infrastructure and distribution networks",
      "Strong cash flow generation in favorable markets"
    ],
    "weaknesses": [
      "Extreme price volatility",
      "High capital expenditure requirements",
      "Environmental concerns affecting public perception",
      "Geopolitical exposure in production regions"
    ],
    "opportunities": [
      "Renewable energy transition",
      "Battery storage technology development",
      "Carbon capture and offset innovations",
      "Digital optimization of production and distribution"
    ],
    "threats": [
      "Accelerating shift to renewable energy",
      "Stringent environmental regulations and carbon taxes",
      "Political instability in key production regions",
      "ESG investing reducing capital access"
    ]
  },
  "financial-services": {
    "strengths": [
      "Stable revenue from fees and interest",
      "Diverse product offerings reduce risk",
      "Strong regulatory barriers protect established players",
      "Data-rich industry enabling customer insights"
    ],
    "weaknesses": [
      "Vulnerable to economic downturns and credit cycles",
      "Heavy regulatory oversight increases compliance costs",
      "Low interest rate environments compress margins",
      "Legacy technology systems require costly updates"
    ],
    "opportunities": [
      "Fintech innovations transforming traditional services",
      "Wealth management for aging populations",
      "Emerging market expansion",
      "Blockchain and digital payment solutions"
    ],
    "threats": [
      "Fintech disruption from non-traditional competitors",
      "Cybersecurity risks and data breaches",
      "Regulatory changes affecting capital requirements",
      "Low interest rate environments"
    ]
  },
  "healthcare": {
    "strengths": [
      "Recession-resistant demand for essential services",
      "Strong patent protections for pharmaceuticals",
      "Aging global population ensures growing demand",
      "High barriers to entry in specialized fields"
    ],
    "weaknesses": [
      "Lengthy and expensive R&D processes",
      "Complex regulatory approval processes",
      "Reliance on insurance reimbursement models",
      "Patent cliffs lead to revenue drops"
    ],
    "opportunities": [
      "Telehealth and digital health expansion",
      "Personalized medicine using genetic data",
      "Emerging market healthcare infrastructure development",
      "AI applications for drug discovery and diagnostics"
    ],
    "threats": [
      "Healthcare reform and pricing pressure",
      "Generic competition after patent expiration",
      "International pricing regulations",
      "Litigation risks for pharmaceuticals and medical devices"
    ]
  },
  "industrials": {
    "strengths": [
      "Diverse revenue streams across multiple industries",
      "Strong intellectual property portfolios",
      "Established customer relationships with high switching costs",
      "Recurring revenue from maintenance and service contracts"
    ],
    "weaknesses": [
      "Cyclical performance tied to economic conditions",
      "High capital expenditure requirements",
      "Exposure to commodity price fluctuations",
      "Labor-intensive processes subject to wage pressures"
    ],
    "opportunities": [
      "Automation and IoT integration",
      "Infrastructure spending initiatives",
      "Sustainable manufacturing practices",
      "Supply chain regionalization"
    ],
    "threats": [
      "Global economic slowdowns affecting capital expenditures",
      "Supply chain disruptions",
      "Labor shortages and wage inflation",
      "Environmental regulations increasing compliance costs"
    ]
  },
  "real-estate": {
    "strengths": [
      "Tangible assets providing inflation hedge",
      "Stable income through long-term leases",
      "Tax advantages through REIT structure",
      "Portfolio diversification across property types"
    ],
    "weaknesses": [
      "Interest rate sensitivity affects financing costs",
      "High capital requirements for acquisitions",
      "Illiquidity of physical assets",
      "Location-dependent performance"
    ],
    "opportunities": [
      "Repurposing retail spaces for mixed use",
      "Data center and logistics property demand",
      "ESG-focused building improvements",
      "Proptech innovations improving efficiency"
    ],
    "threats": [
      "Rising interest rates increasing borrowing costs",
      "Remote work reducing office space demand",
      "E-commerce disrupting traditional retail spaces",
      "Climate change risks to coastal properties"
    ]
  },
  "technology": {
    "strengths": [
      "High profit margins on software products",
      "Recurring revenue through subscription models",
      "Network effects create competitive moats",
      "Scalable business models with low marginal costs"
    ],
    "weaknesses": [
      "Rapid product obsolescence",
      "High R&D expenses required to stay competitive",
      "Talent acquisition and retention challenges",
      "Vulnerability to disruption from new technologies"
    ],
    "opportunities": [
      "AI and machine learning applications",
      "Cloud computing market expansion",
      "Edge computing and IoT integration",
      "Cybersecurity demand growth"
    ],
    "threats": [
      "Increasing regulatory scrutiny and antitrust concerns",
      "Cybersecurity vulnerabilities and attacks",
      "Rapid technological change creating obsolescence risk",
      "Global technology regulation fragmentation"
    ]
  },
  "utilities": {
    "strengths": [
      "Stable and predictable cash flows",
      "Natural monopolies in service territories",
      "Regulatory protection from competition",
      "Essential services with inelastic demand"
    ],
    "weaknesses": [
      "Heavy regulation limiting pricing flexibility",
      "High capital expenditure requirements",
      "Environmental compliance costs",
      "Limited growth potential in mature markets"
    ],
    "opportunities": [
      "Renewable energy transition",
      "Grid modernization initiatives",
      "Electric vehicle charging infrastructure development",
      "Smart meter and grid technology implementation"
    ],
    "threats": [
      "Increasing renewable adoption reducing traditional demand",
      "Rising environmental regulations and compliance costs",
      "Extreme weather events damaging infrastructure",
      "Distributed energy sources bypassing utility providers"
    ]
  }
}; 