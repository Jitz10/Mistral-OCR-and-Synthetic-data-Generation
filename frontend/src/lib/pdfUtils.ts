import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
import autoTable from 'jspdf-autotable';

// API base URL
const API_BASE_URL = 'http://localhost:9000';

/**
 * Fetch detailed stock report data from the backend
 * @param {string} symbol - Stock symbol
 * @returns {Promise<object>} - Stock report data
 */
export const fetchStockReport = async (symbol) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate_stock_report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symbol }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch stock report');
    }

    return response.json();
  } catch (error) {
    console.error('Error fetching stock report:', error);
    return null;
  }
};

/**
 * Format currency values based on stock origin
 * @param {number} value - Value to format
 * @param {boolean} isIndian - Whether it's an Indian stock
 * @returns {string} - Formatted currency string
 */
const formatCurrency = (value, isIndian = false) => {
  if (!value && value !== 0) return 'N/A';
  
  if (isIndian) {
    // Format as Indian currency (₹) in crores
    return `₹${value.toLocaleString('en-IN')} Cr`;
  } else {
    // Format as US currency ($) in millions
    return `$${value.toLocaleString('en-US')} M`;
  }
};

/**
 * Generate PDF from stock report data
 * @param {object} reportData - Stock report data
 * @returns {Promise<void>}
 */
export const generateStockPDF = async (reportData) => {
  if (!reportData) {
    throw new Error('No report data available');
  }

  const {
    symbol,
    companyName,
    reportDate,
    overview,
    fundamentals,
    valuation,
    technicals,
    financials,
    newsAnalysis,
    aiInsights,
    peerComparison,
  } = reportData;

  // Determine if it's an Indian stock
  const isIndianStock = symbol.endsWith('.NS') || symbol.endsWith('.BO') || symbol.endsWith('.BSE');

  // Initialize PDF document
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  // Add title and header
  doc.setFontSize(20);
  doc.setTextColor(0, 51, 102);
  doc.text(`${companyName} (${symbol})`, 105, 15, { align: 'center' });
  
  doc.setFontSize(12);
  doc.setTextColor(100, 100, 100);
  doc.text(`Investment Analysis Report`, 105, 22, { align: 'center' });
  doc.text(`Generated on: ${reportDate}`, 105, 28, { align: 'center' });

  // Add divider
  doc.setDrawColor(200, 200, 200);
  doc.line(20, 32, 190, 32);

  let yPos = 40;

  // Company Overview Section
  doc.setFontSize(14);
  doc.setTextColor(0, 51, 102);
  doc.text('Company Overview', 20, yPos);
  yPos += 8;

  doc.setFontSize(10);
  doc.setTextColor(60, 60, 60);
  
  // Company details table
  const overviewData = [
    ['Exchange', overview.exchange || 'N/A'],
    ['Sector', overview.sector || 'N/A'],
    ['Industry', overview.industry || 'N/A'],
    ['Current Price', overview.currentPrice || 'N/A'],
    ['Market Cap', formatCurrency(overview.marketCap, isIndianStock)],
    ['52 Week High/Low', overview.highLow || 'N/A'],
  ];

  const overviewTable = autoTable(doc, {
    startY: yPos,
    head: [['Parameter', 'Value']],
    body: overviewData,
    theme: 'grid',
    headStyles: { fillColor: [0, 51, 102], textColor: [255, 255, 255] },
    styles: { fontSize: 10 },
    margin: { left: 20, right: 20 },
  });

  yPos = doc.lastAutoTable.finalY + 10;

  // Description
  if (overview.description) {
    doc.setFontSize(11);
    doc.text('Business Description:', 20, yPos);
    yPos += 6;
    
    // Wrap text for description
    const splitText = doc.splitTextToSize(overview.description, 170);
    doc.setFontSize(9);
    doc.text(splitText, 20, yPos);
    yPos += splitText.length * 5 + 10;
  }

  // Key Financial Metrics Section
  if (yPos > 250) { // Add new page if not enough space
    doc.addPage();
    yPos = 20;
  }

  doc.setFontSize(14);
  doc.setTextColor(0, 51, 102);
  doc.text('Key Financial Metrics', 20, yPos);
  yPos += 8;

  // Financials table
  const financialsData = [
    ['Revenue (TTM)', formatCurrency(financials.revenue, isIndianStock)],
    ['Net Income (TTM)', formatCurrency(financials.netIncome, isIndianStock)],
    ['EPS (TTM)', financials.eps || 'N/A'],
    ['Profit Margin', financials.profitMargin ? `${financials.profitMargin}%` : 'N/A'],
    ['ROE', financials.roe ? `${financials.roe}%` : 'N/A'],
    ['Debt to Equity', financials.debtToEquity || 'N/A'],
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Metric', 'Value']],
    body: financialsData,
    theme: 'grid',
    headStyles: { fillColor: [0, 51, 102], textColor: [255, 255, 255] },
    styles: { fontSize: 10 },
    margin: { left: 20, right: 20 },
  });

  yPos = doc.lastAutoTable.finalY + 10;

  // Valuation Metrics Section
  if (yPos > 220) { // Add new page if not enough space
    doc.addPage();
    yPos = 20;
  }

  doc.setFontSize(14);
  doc.setTextColor(0, 51, 102);
  doc.text('Valuation Metrics', 20, yPos);
  yPos += 8;

  // Valuation table
  const valuationData = [
    ['P/E Ratio', valuation.pe || 'N/A'],
    ['Forward P/E', valuation.forwardPE || 'N/A'],
    ['PEG Ratio', valuation.pegRatio || 'N/A'],
    ['Price to Book', valuation.priceToBook || 'N/A'],
    ['Price to Sales', valuation.priceToSales || 'N/A'],
    ['EV/EBITDA', valuation.evToEbitda || 'N/A'],
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Metric', 'Value']],
    body: valuationData,
    theme: 'grid',
    headStyles: { fillColor: [0, 51, 102], textColor: [255, 255, 255] },
    styles: { fontSize: 10 },
    margin: { left: 20, right: 20 },
  });

  yPos = doc.lastAutoTable.finalY + 10;

  // Technical Analysis Section
  if (yPos > 220) { // Add new page if not enough space
    doc.addPage();
    yPos = 20;
  }

  doc.setFontSize(14);
  doc.setTextColor(0, 51, 102);
  doc.text('Technical Analysis', 20, yPos);
  yPos += 8;

  // Technical indicators table
  const technicalsData = [
    ['RSI (14)', technicals.rsi || 'N/A'],
    ['MACD', technicals.macd || 'N/A'],
    ['Moving Avg (50)', technicals.ma50 || 'N/A'],
    ['Moving Avg (200)', technicals.ma200 || 'N/A'],
    ['Volume (Avg)', technicals.avgVolume ? technicals.avgVolume.toLocaleString() : 'N/A'],
    ['Beta', technicals.beta || 'N/A'],
  ];

  autoTable(doc, {
    startY: yPos,
    head: [['Indicator', 'Value']],
    body: technicalsData,
    theme: 'grid',
    headStyles: { fillColor: [0, 51, 102], textColor: [255, 255, 255] },
    styles: { fontSize: 10 },
    margin: { left: 20, right: 20 },
  });

  yPos = doc.lastAutoTable.finalY + 10;

  // AI Insights Section - add new page
  doc.addPage();
  yPos = 20;

  doc.setFontSize(14);
  doc.setTextColor(0, 51, 102);
  doc.text('AI-Generated Investment Insights', 20, yPos);
  yPos += 8;

  doc.setFontSize(11);
  doc.setTextColor(0, 0, 0);
  
  // Executive Summary
  doc.text('Executive Summary:', 20, yPos);
  yPos += 6;
  
  const summaryText = doc.splitTextToSize(aiInsights.summary || 'No AI insights available.', 170);
  doc.setFontSize(10);
  doc.text(summaryText, 20, yPos);
  yPos += summaryText.length * 5 + 8;

  // Strengths
  if (aiInsights.strengths && aiInsights.strengths.length > 0) {
    doc.setFontSize(11);
    doc.text('Key Strengths:', 20, yPos);
    yPos += 6;
    
    doc.setFontSize(10);
    aiInsights.strengths.forEach(strength => {
      const strengthText = doc.splitTextToSize(`• ${strength}`, 170);
      doc.text(strengthText, 20, yPos);
      yPos += strengthText.length * 5 + 2;
    });
    yPos += 4;
  }

  // Risks
  if (aiInsights.risks && aiInsights.risks.length > 0) {
    doc.setFontSize(11);
    doc.text('Key Risks:', 20, yPos);
    yPos += 6;
    
    doc.setFontSize(10);
    aiInsights.risks.forEach(risk => {
      const riskText = doc.splitTextToSize(`• ${risk}`, 170);
      doc.text(riskText, 20, yPos);
      yPos += riskText.length * 5 + 2;
    });
    yPos += 4;
  }

  // Investment Recommendation
  if (yPos > 240) { // Add new page if not enough space
    doc.addPage();
    yPos = 20;
  }

  doc.setFontSize(11);
  doc.text('Investment Recommendation:', 20, yPos);
  yPos += 6;
  
  const recommendationText = doc.splitTextToSize(aiInsights.recommendation || 'No recommendation available.', 170);
  doc.setFontSize(10);
  doc.setTextColor(0, 0, 0);
  doc.text(recommendationText, 20, yPos);
  yPos += recommendationText.length * 5 + 8;

  // Peer Comparison Section
  if (yPos > 200 || !peerComparison || peerComparison.length === 0) { // Add new page
    doc.addPage();
    yPos = 20;
  }

  if (peerComparison && peerComparison.length > 0) {
    doc.setFontSize(14);
    doc.setTextColor(0, 51, 102);
    doc.text('Peer Comparison', 20, yPos);
    yPos += 8;

    // Prepare peer comparison data
    const peerHeaders = ['Company', 'Price', 'P/E', 'Market Cap', 'ROE'];
    
    const peerRows = peerComparison.map(peer => [
      peer.name || 'N/A',
      peer.price || 'N/A',
      peer.pe || 'N/A',
      formatCurrency(peer.marketCap, isIndianStock),
      peer.roe ? `${peer.roe}%` : 'N/A'
    ]);

    autoTable(doc, {
      startY: yPos,
      head: [peerHeaders],
      body: peerRows,
      theme: 'grid',
      headStyles: { fillColor: [0, 51, 102], textColor: [255, 255, 255] },
      styles: { fontSize: 9, cellPadding: 2 },
      margin: { left: 20, right: 20 },
    });

    yPos = doc.lastAutoTable.finalY + 10;
  }

  // Recent News Analysis
  if (newsAnalysis && newsAnalysis.length > 0) {
    if (yPos > 180) { // Add new page if not enough space
      doc.addPage();
      yPos = 20;
    }

    doc.setFontSize(14);
    doc.setTextColor(0, 51, 102);
    doc.text('Recent News Analysis', 20, yPos);
    yPos += 8;

    doc.setFontSize(10);
    newsAnalysis.slice(0, 5).forEach((news, index) => {
      doc.setTextColor(0, 0, 0);
      const newsTitle = doc.splitTextToSize(`${index + 1}. ${news.title}`, 170);
      doc.text(newsTitle, 20, yPos);
      yPos += newsTitle.length * 5 + 2;

      // News date and sentiment
      doc.setFontSize(9);
      doc.setTextColor(100, 100, 100);
      doc.text(`Date: ${news.date} | Sentiment: ${news.sentiment}`, 20, yPos);
      yPos += 5;

      // News summary if available
      if (news.summary) {
        const newsSummary = doc.splitTextToSize(news.summary, 170);
        doc.setFontSize(8);
        doc.text(newsSummary, 20, yPos);
        yPos += newsSummary.length * 4 + 6;
      } else {
        yPos += 6;
      }
    });
  }

  // Footer
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Add page number
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(`Page ${i} of ${pageCount}`, 105, 290, { align: 'center' });
    
    // Add disclaimer
    if (i === pageCount) {
      doc.text('Disclaimer: This report is generated using AI and should not be considered as financial advice. Always conduct your own research before making investment decisions.', 105, 280, { align: 'center', maxWidth: 170 });
    }
  }

  // Save the PDF
  doc.save(`${symbol}_Investment_Analysis_${new Date().toISOString().split('T')[0]}.pdf`);
};