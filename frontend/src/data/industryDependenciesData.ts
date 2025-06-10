export interface DependencyData {
  sector: string;
  impact: 'Critical' | 'High' | 'Medium' | 'Low';
  description: string;
}

export interface SectorDependencies {
  [key: string]: DependencyData[];
}

export const industryDependenciesData: SectorDependencies = {
  "basic-materials": [
    { 
      sector: "Energy", 
      impact: "Critical", 
      description: "Energy costs represent significant portion of production expenses, with energy-intensive processes requiring stable and cost-effective power sources." 
    },
    { 
      sector: "Manufacturing", 
      impact: "Medium", 
      description: "Manufacturing sector consumes various materials for production processes, with demand tied to industrial output and capacity utilization." 
    },
    { 
      sector: "Transportation", 
      impact: "High", 
      description: "Logistics and transportation networks crucial for raw material delivery and finished product distribution, with costs affected by fuel prices and infrastructure." 
    },
    { 
      sector: "Construction", 
      impact: "High", 
      description: "Construction industry demand drives significant portion of materials consumption, particularly in infrastructure and commercial building projects." 
    }
  ],
  "communication-services": [
    { 
      sector: "Media", 
      impact: "High", 
      description: "Content creation and distribution partnerships essential for service offerings, with streaming and digital media driving bandwidth demand." 
    },
    { 
      sector: "Consumer Electronics", 
      impact: "High", 
      description: "Device ecosystem and compatibility crucial for service adoption, with smartphone and smart device penetration driving subscriber growth." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Government policies and spectrum allocation affect service deployment and pricing strategies across different markets." 
    },
    { 
      sector: "Technology", 
      impact: "Critical", 
      description: "Technology infrastructure and innovation drive service capabilities, with 5G and fiber optic networks requiring significant capital investment." 
    }
  ],
  "consumer-cyclical": [
    { 
      sector: "Retail", 
      impact: "Critical", 
      description: "Retail channel performance and consumer spending patterns directly affect sales volumes and inventory management." 
    },
    { 
      sector: "E-commerce", 
      impact: "High", 
      description: "Digital sales channels and fulfillment networks increasingly important for reaching consumers and managing distribution." 
    },
    { 
      sector: "Advertising", 
      impact: "High", 
      description: "Marketing and brand awareness crucial for driving consumer demand and maintaining market share." 
    },
    { 
      sector: "Supply Chain", 
      impact: "Medium", 
      description: "Global supply chain efficiency affects product availability and cost structure, particularly for imported goods." 
    }
  ],
  "consumer-defensive": [
    { 
      sector: "Agriculture", 
      impact: "Critical", 
      description: "Raw material sourcing and commodity prices significantly impact production costs and margin stability." 
    },
    { 
      sector: "Retail", 
      impact: "Medium", 
      description: "Distribution channels and shelf space allocation crucial for product visibility and sales performance." 
    },
    { 
      sector: "Packaging", 
      impact: "High", 
      description: "Packaging materials and design affect product preservation, transportation efficiency, and consumer appeal." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Food safety and labeling regulations impact product formulation and marketing strategies." 
    }
  ],
  "energy": [
    { 
      sector: "Transportation", 
      impact: "Critical", 
      description: "Transportation sector accounts for significant portion of energy demand, particularly for refined products and natural gas." 
    },
    { 
      sector: "Industrial", 
      impact: "Critical", 
      description: "Industrial energy consumption drives demand for power generation and process fuels across manufacturing sectors." 
    },
    { 
      sector: "Technology", 
      impact: "High", 
      description: "Digital infrastructure and monitoring systems crucial for production optimization and distribution efficiency." 
    },
    { 
      sector: "Environmental", 
      impact: "Medium", 
      description: "Environmental regulations and carbon pricing mechanisms affect production costs and investment decisions." 
    }
  ],
  "financial-services": [
    
    { 
      sector: "Real Estate", 
      impact: "High", 
      description: "Real estate market conditions affect loan portfolio performance and mortgage banking revenue." 
    },
    { 
      sector: "Technology", 
      impact: "Critical", 
      description: "Digital infrastructure and cybersecurity systems essential for service delivery and data protection." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Financial regulations and compliance requirements impact operational costs and business strategies." 
    },
    { 
      sector: "Consumer", 
      impact: "Medium", 
      description: "Consumer spending patterns and credit behavior influence loan demand and credit quality." 
    }
  ],
  "healthcare": [
    { 
      sector: "Technology", 
      impact: "Critical", 
      description: "Medical technology and digital health solutions driving innovation in patient care and operational efficiency." 
    },
    { 
      sector: "Pharmaceutical", 
      impact: "High", 
      description: "Drug development and pricing strategies significantly impact treatment costs and reimbursement models." 
    },
    { 
      sector: "Insurance", 
      impact: "Medium", 
      description: "Insurance coverage and reimbursement policies affect patient access and provider revenue streams." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Healthcare regulations and compliance requirements impact service delivery and operational costs." 
    }
  ],
  "industrials": [
    { 
      sector: "Technology", 
      impact: "Critical", 
      description: "Automation and digital solutions driving efficiency improvements and new service offerings." 
    },
    { 
      sector: "Energy", 
      impact: "High", 
      description: "Energy costs and availability crucial for manufacturing processes and operational efficiency." 
    },
    { 
      sector: "Transportation", 
      impact: "High", 
      description: "Logistics networks and transportation infrastructure essential for supply chain management." 
    },
    { 
      sector: "Construction", 
      impact: "Medium", 
      description: "Construction activity levels drive demand for industrial equipment and materials." 
    }
  ],
  "real-estate": [
    { 
      sector: "Technology", 
      impact: "High", 
      description: "Proptech solutions transforming property management and tenant experiences." 
    },
    { 
      sector: "Financial Services", 
      impact: "Critical", 
      description: "Mortgage availability and interest rates significantly impact property demand and valuations." 
    },
    { 
      sector: "Construction", 
      impact: "High", 
      description: "Construction costs and material availability affect development feasibility and project timelines." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Zoning laws and building codes impact development opportunities and property use." 
    }
  ],
  "technology": [
    { 
      sector: "Semiconductors", 
      impact: "Critical", 
      description: "Semiconductor supply chain crucial for hardware production and component availability." 
    },
    { 
      sector: "Regulatory", 
      impact: "Medium", 
      description: "Data privacy regulations and technology standards affect product development and market access." 
    },
    { 
      sector: "Cloud Infrastructure", 
      impact: "High", 
      description: "Cloud service providers essential for software deployment and data management." 
    },
    { 
      sector: "Cybersecurity", 
      impact: "High", 
      description: "Security solutions and threat protection critical for maintaining system integrity." 
    }
  ],
  "utilities": [
    { 
      sector: "Technology", 
      impact: "High", 
      description: "Smart grid technology and monitoring systems crucial for operational efficiency." 
    },
    { 
      sector: "Regulatory", 
      impact: "High", 
      description: "Rate regulations and environmental policies significantly impact business operations." 
    },
    { 
      sector: "Energy", 
      impact: "Critical", 
      description: "Energy generation and distribution infrastructure essential for service reliability." 
    },
    { 
      sector: "Construction", 
      impact: "Medium", 
      description: "Infrastructure development and maintenance requirements affect capital expenditure planning." 
    }
  ]
}; 