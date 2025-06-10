import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { FiMenu, FiHome, FiBarChart2, FiPieChart, FiDollarSign, FiUsers, FiGlobe, FiTrendingUp, FiGrid, FiBriefcase, FiMessageSquare } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend);

const sectionOrder = [
  { key: 'about_company', label: 'About Company', icon: <FiHome /> },
  { key: 'about_sector', label: 'About Sector', icon: <FiGlobe /> },
  { key: 'key_ratios', label: 'Key Ratios', icon: <FiBarChart2 /> },
  { key: 'profit_and_loss_statement', label: 'Profit & Loss', icon: <FiDollarSign /> },
  { key: 'balance_sheet', label: 'Balance Sheet', icon: <FiGrid /> },
  { key: 'cashflow_statement', label: 'Cash Flow', icon: <FiTrendingUp /> },
  { key: 'predictions', label: 'Predictions', icon: <FiPieChart /> },
  { key: 'competition', label: 'Competition', icon: <FiBriefcase /> },
  { key: 'shareholders_information', label: 'Shareholders', icon: <FiUsers /> },
  { key: 'concall_presentation_summary', label: 'Concall Summary', icon: <FiMessageSquare /> },
];

// Color Palette
const colors = {
  primary: '#1E40AF', // Deep blue
  secondary: '#3B82F6', // Bright blue
  accent: '#60A5FA', // Light blue
  light: {
    background: '#F8FAFC',
    text: '#1E293B',
    card: '#FFFFFF',
    border: '#E2E8F0',
  },
  dark: {
    background: '#1E293B',
    text: '#F8FAFC',
    card: '#334155',
    border: '#475569',
  },
  sidebar: {
    background: '#1E40AF',
    text: '#FFFFFF',
  },
};

// Error Boundary for Charts
class ChartErrorBoundary extends React.Component {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      return <p className="text-red-600 text-sm">Failed to render chart.</p>;
    }
    return this.props.children;
  }
}

// Reusable Components
const KeyRatios = ({ ratios, theme }) => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
    {Object.entries(ratios).map(([key, value]) => (
      <Card key={key} style={{
        position: 'relative',
        borderLeft: '4px solid #3B82F6',
        borderRadius: '8px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        transition: 'all 0.3s ease-in-out',
        backgroundColor: 'white'
      }}>
        <CardContent className="p-4">
          <p style={{
            fontSize: '0.875rem',
            fontWeight: 500,
            color: '#64748B',
            textTransform: 'capitalize',
            marginBottom: '0.5rem'
          }}>
            {key.replace(/_/g, ' ')}
          </p>
          <p style={{
            fontSize: '1.25rem',
            fontWeight: 600,
            color: '#3B82F6'
          }}>
            {value}
          </p>
        </CardContent>
      </Card>
    ))}
  </div>
);

const DataTable = ({ items, columns, title, theme }) => (
  <Card className={`shadow-md transition-all duration-300 bg-${theme.card} border-${theme.border}`} style={{
    position: 'relative',
    borderLeft: '4px solid #1E40AF',
    borderRadius: '8px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
  }}>
    <CardHeader style={{
      borderBottom: '1px solid #E5E7EB',
      backgroundColor: '#F8FAFC',
      borderTopLeftRadius: '8px',
      borderTopRightRadius: '8px'
    }}>
      <CardTitle className={`text-lg font-semibold text-${theme.text}`} style={{
        color: '#1E40AF',
        fontSize: '1.125rem',
        fontWeight: 600
      }}>
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-sm" style={{
          borderSpacing: 0,
          borderRadius: '8px',
          overflow: 'hidden'
        }}>
          <thead>
            <tr style={{
              backgroundColor: '#1E40AF',
              color: 'white'
            }}>
              {columns.map((col) => (
                <th 
                  key={col} 
                  className="px-4 py-3 text-left font-medium capitalize"
                  style={{
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    letterSpacing: '0.05em',
                    borderBottom: '2px solid #1E3A8A'
                  }}
                >
                  {col.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((item, index) => (
              <tr 
                key={index} 
                style={{
                  backgroundColor: index % 2 === 0 ? 'white' : '#F8FAFC',
                  transition: 'all 0.2s ease-in-out'
                }}
                className="hover:bg-blue-50"
              >
                {columns.map((col) => (
                  <td 
                    key={col} 
                    className="px-4 py-3"
                    style={{
                      fontSize: '0.875rem',
                      color: '#1F2937',
                      borderBottom: '1px solid #E5E7EB'
                    }}
                  >
                    {col === 'beneficiaries' || col === 'competitors' ? (
                      Array.isArray(item[col]) ? item[col].join(', ') : item[col]
                    ) : (
                      item[col] || '-'
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </CardContent>
  </Card>
);

const ProfitLossChart = ({ data, predictions, theme }) => {
  const plData = data?.periods || [];
  const projData = predictions?.profit_and_loss_statement?.data || {};
  const labels = [...plData.map(p => p.period), 'Mar-25 (Proj)'];
  const sales = [...plData.map(p => p.sales), projData.sales || 0];
  const operatingProfit = [...plData.map(p => p.operating_profit), projData.operating_profit || 0];
  const netProfit = [...plData.map(p => p.net_profit), projData.net_profit || 0];

  return (
    <Card style={{
      position: 'relative',
      borderLeft: '4px solid #60A5FA',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      backgroundColor: 'white'
    }}>
      <CardHeader style={{
        borderBottom: '1px solid #E5E7EB',
        backgroundColor: '#F8FAFC',
        borderTopLeftRadius: '8px',
        borderTopRightRadius: '8px'
      }}>
        <CardTitle style={{
          color: '#60A5FA',
          fontSize: '1.125rem',
          fontWeight: 600
        }}>
          Profit & Loss Trends
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartErrorBoundary>
          <Bar
            data={{
              labels,
              datasets: [
                { label: 'Sales (₹ Cr)', data: sales, backgroundColor: '#1E40AF', hoverBackgroundColor: '#1E3A8A' },
                { label: 'Operating Profit (₹ Cr)', data: operatingProfit, backgroundColor: '#3B82F6', hoverBackgroundColor: '#2563EB' },
                { label: 'Net Profit (₹ Cr)', data: netProfit, backgroundColor: '#60A5FA', hoverBackgroundColor: '#3B82F6' },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Amount (₹ Cr)', font: { size: 12 }, color: theme.text } },
                x: { title: { display: true, text: 'Fiscal Year', font: { size: 12 }, color: theme.text } },
              },
              plugins: {
                legend: { position: 'top', labels: { font: { size: 12 }, color: theme.text } },
                tooltip: { backgroundColor: theme.background, titleColor: theme.text, bodyColor: theme.text },
              },
              animation: { duration: 1000, easing: 'easeOutQuart' },
            }}
            height={window.innerWidth < 640 ? 200 : 300}
          />
        </ChartErrorBoundary>
      </CardContent>
    </Card>
  );
};

const ShareholdingChart = ({ data, theme }) => {
  const shareData = data?.shareholding_pattern || [];
  const latest = shareData[shareData.length - 1] || {};
  const labels = ['Promoter', 'DII', 'FII', 'Others'];
  const values = [latest.promoter, latest.dii, latest.fii, latest.others].map(v => parseFloat(v) || 0);

  return (
    <Card style={{
      position: 'relative',
      borderLeft: '4px solid #93C5FD',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      backgroundColor: 'white'
    }}>
      <CardHeader style={{
        borderBottom: '1px solid #E5E7EB',
        backgroundColor: '#F8FAFC',
        borderTopLeftRadius: '8px',
        borderTopRightRadius: '8px'
      }}>
        <CardTitle style={{
          color: '#93C5FD',
          fontSize: '1.125rem',
          fontWeight: 600
        }}>
          Shareholding Pattern (Jun-24)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartErrorBoundary>
          <Pie
            data={{
              labels,
              datasets: [{
                data: values,
                backgroundColor: ['#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD'],
                hoverBackgroundColor: ['#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA'],
              }],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'right', labels: { font: { size: 12 }, color: theme.text } },
                tooltip: { backgroundColor: theme.background, titleColor: theme.text, bodyColor: theme.text },
              },
              animation: { duration: 1000, easing: 'easeOutQuart' },
            }}
            height={200}
          />
        </ChartErrorBoundary>
      </CardContent>
    </Card>
  );
};

const CashFlowChart = ({ data, theme }) => {
  const cfData = data?.periods || [];
  const labels = cfData.map(p => p.period);
  const operating = cfData.map(p => p.cash_from_operating_activity);
  const investing = cfData.map(p => p.cash_from_investing_activity);
  const financing = cfData.map(p => p.cash_from_financing_activity);

  return (
    <Card style={{
      position: 'relative',
      borderLeft: '4px solid #BFDBFE',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      backgroundColor: 'white'
    }}>
      <CardHeader style={{
        borderBottom: '1px solid #E5E7EB',
        backgroundColor: '#F8FAFC',
        borderTopLeftRadius: '8px',
        borderTopRightRadius: '8px'
      }}>
        <CardTitle style={{
          color: '#BFDBFE',
          fontSize: '1.125rem',
          fontWeight: 600
        }}>
          Cash Flow Trends
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartErrorBoundary>
          <Line
            data={{
              labels,
              datasets: [
                { label: 'Operating Cash Flow', data: operating, borderColor: '#1E40AF', fill: false },
                { label: 'Investing Cash Flow', data: investing, borderColor: '#3B82F6', fill: false },
                { label: 'Financing Cash Flow', data: financing, borderColor: '#60A5FA', fill: false },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                y: { title: { display: true, text: 'Amount (₹ Cr)', font: { size: 12 }, color: theme.text } },
                x: { title: { display: true, text: 'Fiscal Year', font: { size: 12 }, color: theme.text } },
              },
              plugins: {
                legend: { position: 'top', labels: { font: { size: 12 }, color: theme.text } },
                tooltip: { backgroundColor: theme.background, titleColor: theme.text, bodyColor: theme.text },
              },
              animation: { duration: 1000, easing: 'easeOutQuart' },
            }}
            height={window.innerWidth < 640 ? 200 : 300}
          />
        </ChartErrorBoundary>
      </CardContent>
    </Card>
  );
};

// Top Navigation Component
const TopNav = ({ activeSection, setActiveSection }) => {
  const navigate = useNavigate();

  return (
    <nav style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 50,
      background: 'linear-gradient(to right, #1E40AF, #2563EB)',
      color: 'white',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
    }}>
      <div style={{
        maxWidth: '1280px',
        margin: '0 auto'
      }}>
        {/* Top Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '1rem 1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <h2 
              onClick={() => navigate('/')}
              style={{
                fontSize: '1.5rem',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                cursor: 'pointer',
                transition: 'opacity 0.2s ease-in-out',
                '&:hover': {
                  opacity: 0.8
                }
              }}
            >
              <span style={{ color: '#60A5FA' }}>Quant</span>
              <span style={{ color: 'white' }}>AI</span>
            </h2>
          </div>
        </div>

        {/* Navigation Items */}
        <div style={{
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '0.5rem 1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            overflowX: 'auto',
            padding: '0.5rem 0',
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
            '&::-webkit-scrollbar': {
              display: 'none'
            }
          }}>
            {sectionOrder.map(({ key, label, icon }) => (
              <button
                key={key}
                onClick={() => setActiveSection(key)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '0.75rem 1rem',
                  borderRadius: '0.5rem',
                  transition: 'all 0.2s ease-in-out',
                  whiteSpace: 'nowrap',
                  background: activeSection === key ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
                  color: activeSection === key ? 'white' : 'rgba(255, 255, 255, 0.8)',
                  border: 'none',
                  cursor: 'pointer',
                  '&:hover': {
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: 'white'
                  }
                }}
                aria-label={label}
              >
                <span style={{ marginRight: '0.5rem' }}>{icon}</span>
                <span style={{ fontSize: '0.875rem' }}>{label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

// Header Component
const Header = ({ companyName }) => (
  <header style={{
    background: 'linear-gradient(135deg, #8B5CF6, #EC4899)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
  }}>
    <div style={{
      maxWidth: '1280px',
      margin: '0 auto',
      padding: '2rem 1.5rem'
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        <h1 style={{
          fontSize: '2rem',
          fontWeight: '800',
          color: 'white',
          margin: 0,
          textShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          letterSpacing: '-0.025em',
          background: 'linear-gradient(to right, #FFFFFF, #F3F4F6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          display: 'inline-block'
        }}>
          {companyName}
        </h1>
        <p style={{
          fontSize: '1.125rem',
          color: 'rgba(255, 255, 255, 0.9)',
          margin: 0,
          fontWeight: '500'
        }}>
          Financial Analytics Dashboard
        </p>
      </div>
    </div>
  </header>
);

// Section Content Component
const SectionContent = ({ activeSection, filteredData, theme }) => {
  const value = filteredData[activeSection];
  if (!value) {
    return (
      <Card className={`shadow-md bg-${theme.card} border-${theme.border}`}>
        <CardContent className="p-6">
          <p className={`text-sm text-${theme.text}`}>No data available</p>
        </CardContent>
      </Card>
    );
  }

  switch (activeSection) {
    case 'about_company':
      return (
        <Card style={{
          position: 'relative',
          borderLeft: '4px solid #1E40AF',
          borderRadius: '8px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          backgroundColor: 'white'
        }}>
          <CardHeader style={{
            borderBottom: '1px solid #E5E7EB',
            backgroundColor: '#F8FAFC',
            borderTopLeftRadius: '8px',
            borderTopRightRadius: '8px'
          }}>
            <CardTitle style={{
              color: '#1E40AF',
              fontSize: '1.125rem',
              fontWeight: 600
            }}>
              About Company
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <p className="text-base leading-relaxed text-gray-700">
              {value.description}
            </p>
          </CardContent>
        </Card>
      );
    case 'about_sector':
      return (
        <Card style={{
          position: 'relative',
          borderLeft: '4px solid #3B82F6',
          borderRadius: '8px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          backgroundColor: 'white'
        }}>
          <CardHeader style={{
            borderBottom: '1px solid #E5E7EB',
            backgroundColor: '#F8FAFC',
            borderTopLeftRadius: '8px',
            borderTopRightRadius: '8px'
          }}>
            <CardTitle style={{
              color: '#3B82F6',
              fontSize: '1.125rem',
              fontWeight: 600
            }}>
              About Sector
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <p className="text-base leading-relaxed text-gray-700">
              {value.description}
            </p>
          </CardContent>
        </Card>
      );
    case 'concall_presentation_summary':
      return (
        <Card style={{
          position: 'relative',
          borderLeft: '4px solid #60A5FA',
          borderRadius: '8px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          backgroundColor: 'white'
        }}>
          <CardHeader style={{
            borderBottom: '1px solid #E5E7EB',
            backgroundColor: '#F8FAFC',
            borderTopLeftRadius: '8px',
            borderTopRightRadius: '8px'
          }}>
            <CardTitle style={{
              color: '#60A5FA',
              fontSize: '1.125rem',
              fontWeight: 600
            }}>
              Concall Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <p className="text-base leading-relaxed text-gray-700">
              {value.description}
            </p>
          </CardContent>
        </Card>
      );
    case 'key_ratios':
      return <KeyRatios ratios={value} theme={theme} />;
    case 'profit_and_loss_statement':
      return (
        <div className="space-y-6">
          <ProfitLossChart data={value} predictions={filteredData.predictions} theme={theme} />
          <DataTable
            items={value.periods}
            columns={['period', 'sales', 'yoy_growth', 'expenses', 'operating_profit', 'opm', 'net_profit', 'npm', 'yoy_growth_net_profit']}
            title="Profit & Loss Statement"
            theme={theme}
          />
        </div>
      );
    case 'balance_sheet':
      return (
        <DataTable
          items={value.periods}
          columns={['period', 'equity_share_capital', 'reserves', 'borrowings', 'yoy_growth_borrowings', 'total_liabilities', 'total_assets', 'receivables', 'inventory', 'cash_and_bank']}
          title="Balance Sheet"
          theme={theme}
        />
      );
    case 'cashflow_statement':
      return (
        <div className="space-y-6">
          <CashFlowChart data={value} theme={theme} />
          <DataTable
            items={value.periods}
            columns={['period', 'cash_from_operating_activity', 'cash_from_investing_activity', 'cash_from_financing_activity', 'net_cash_flow']}
            title="Cash Flow Statement"
            theme={theme}
          />
        </div>
      );
    case 'predictions':
      return (
        <div className="space-y-6">
          {/* Profit & Loss Predictions */}
          <Card style={{
            position: 'relative',
            borderLeft: '4px solid #1E40AF',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            backgroundColor: 'white'
          }}>
            <CardHeader style={{
              borderBottom: '1px solid #E5E7EB',
              backgroundColor: '#F8FAFC',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px'
            }}>
              <CardTitle style={{
                color: '#1E40AF',
                fontSize: '1.125rem',
                fontWeight: 600
              }}>
                Profit & Loss Forecast (2025-2027)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-sm" style={{
                  borderSpacing: 0,
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}>
                  <thead>
                    <tr style={{
                      backgroundColor: '#1E40AF',
                      color: 'white'
                    }}>
                      <th className="px-4 py-3 text-left font-medium">Year</th>
                      <th className="px-4 py-3 text-left font-medium">Sales (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">YoY Growth</th>
                      <th className="px-4 py-3 text-left font-medium">Operating Profit</th>
                      <th className="px-4 py-3 text-left font-medium">OPM</th>
                      <th className="px-4 py-3 text-left font-medium">Net Profit</th>
                      <th className="px-4 py-3 text-left font-medium">NPM</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2025</td>
                      <td className="px-4 py-3">6,015</td>
                      <td className="px-4 py-3">15%</td>
                      <td className="px-4 py-3">420</td>
                      <td className="px-4 py-3">7.0%</td>
                      <td className="px-4 py-3">197</td>
                      <td className="px-4 py-3">3.3%</td>
                    </tr>
                    <tr style={{ backgroundColor: '#F8FAFC' }}>
                      <td className="px-4 py-3">2026</td>
                      <td className="px-4 py-3">6,917</td>
                      <td className="px-4 py-3">15%</td>
                      <td className="px-4 py-3">484</td>
                      <td className="px-4 py-3">7.0%</td>
                      <td className="px-4 py-3">227</td>
                      <td className="px-4 py-3">3.3%</td>
                    </tr>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2027</td>
                      <td className="px-4 py-3">7,955</td>
                      <td className="px-4 py-3">15%</td>
                      <td className="px-4 py-3">557</td>
                      <td className="px-4 py-3">7.0%</td>
                      <td className="px-4 py-3">261</td>
                      <td className="px-4 py-3">3.3%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Balance Sheet Predictions */}
          <Card style={{
            position: 'relative',
            borderLeft: '4px solid #3B82F6',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            backgroundColor: 'white'
          }}>
            <CardHeader style={{
              borderBottom: '1px solid #E5E7EB',
              backgroundColor: '#F8FAFC',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px'
            }}>
              <CardTitle style={{
                color: '#3B82F6',
                fontSize: '1.125rem',
                fontWeight: 600
              }}>
                Balance Sheet Forecast (2025-2027)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-sm" style={{
                  borderSpacing: 0,
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}>
                  <thead>
                    <tr style={{
                      backgroundColor: '#3B82F6',
                      color: 'white'
                    }}>
                      <th className="px-4 py-3 text-left font-medium">Year</th>
                      <th className="px-4 py-3 text-left font-medium">Total Assets (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">Borrowings (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">Reserves (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">Receivables (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">Inventory (₹ Cr)</th>
                      <th className="px-4 py-3 text-left font-medium">Cash & Bank (₹ Cr)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2025</td>
                      <td className="px-4 py-3">5,200</td>
                      <td className="px-4 py-3">220</td>
                      <td className="px-4 py-3">1,500</td>
                      <td className="px-4 py-3">1,700</td>
                      <td className="px-4 py-3">950</td>
                      <td className="px-4 py-3">150</td>
                    </tr>
                    <tr style={{ backgroundColor: '#F8FAFC' }}>
                      <td className="px-4 py-3">2026</td>
                      <td className="px-4 py-3">5,980</td>
                      <td className="px-4 py-3">253</td>
                      <td className="px-4 py-3">1,727</td>
                      <td className="px-4 py-3">1,955</td>
                      <td className="px-4 py-3">1,093</td>
                      <td className="px-4 py-3">173</td>
                    </tr>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2027</td>
                      <td className="px-4 py-3">6,877</td>
                      <td className="px-4 py-3">291</td>
                      <td className="px-4 py-3">1,986</td>
                      <td className="px-4 py-3">2,248</td>
                      <td className="px-4 py-3">1,257</td>
                      <td className="px-4 py-3">198</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Cash Flow Predictions */}
          <Card style={{
            position: 'relative',
            borderLeft: '4px solid #60A5FA',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            backgroundColor: 'white'
          }}>
            <CardHeader style={{
              borderBottom: '1px solid #E5E7EB',
              backgroundColor: '#F8FAFC',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px'
            }}>
              <CardTitle style={{
                color: '#60A5FA',
                fontSize: '1.125rem',
                fontWeight: 600
              }}>
                Cash Flow Forecast (2025-2027)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-sm" style={{
                  borderSpacing: 0,
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}>
                  <thead>
                    <tr style={{
                      backgroundColor: '#60A5FA',
                      color: 'white'
                    }}>
                      <th className="px-4 py-3 text-left font-medium">Year</th>
                      <th className="px-4 py-3 text-left font-medium">Operating Cash Flow</th>
                      <th className="px-4 py-3 text-left font-medium">Investing Cash Flow</th>
                      <th className="px-4 py-3 text-left font-medium">Financing Cash Flow</th>
                      <th className="px-4 py-3 text-left font-medium">Net Cash Flow</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2025</td>
                      <td className="px-4 py-3">300</td>
                      <td className="px-4 py-3">-120</td>
                      <td className="px-4 py-3">-150</td>
                      <td className="px-4 py-3">30</td>
                    </tr>
                    <tr style={{ backgroundColor: '#F8FAFC' }}>
                      <td className="px-4 py-3">2026</td>
                      <td className="px-4 py-3">345</td>
                      <td className="px-4 py-3">-138</td>
                      <td className="px-4 py-3">-173</td>
                      <td className="px-4 py-3">34</td>
                    </tr>
                    <tr style={{ backgroundColor: 'white' }}>
                      <td className="px-4 py-3">2027</td>
                      <td className="px-4 py-3">397</td>
                      <td className="px-4 py-3">-159</td>
                      <td className="px-4 py-3">-198</td>
                      <td className="px-4 py-3">40</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    case 'competition':
      return (
        <div className="space-y-6">
          {value.segments && (
            <DataTable
              items={value.segments}
              columns={['segment', 'voltage_range', 'market_size', 'beneficiaries']}
              title="Market Segments"
              theme={theme}
            />
          )}
          {value.market_by_industry && (
            <DataTable
              items={value.market_by_industry}
              columns={['user_industry', 'voltage_class', 'market_size', 'competitors']}
              title="Market by Industry"
              theme={theme}
            />
          )}
        </div>
      );
    case 'shareholders_information':
      return (
        <div className="space-y-6">
          <ShareholdingChart data={value} theme={theme} />
          <DataTable
            items={value.shareholding_pattern}
            columns={['period', 'promoter', 'dii', 'fii', 'others']}
            title="Shareholding Pattern"
            theme={theme}
          />
        </div>
      );
    default:
      return (
        <Card className={`shadow-md bg-${theme.card} border-${theme.border}`}>
          <CardContent className="p-6">
            {typeof value === 'object' && value.description ? (
              <p className={`text-base leading-relaxed text-${theme.text}`}>
                {value.description}
              </p>
            ) : (
              <pre className={`text-sm text-${theme.text}`}>
                {JSON.stringify(value, null, 2)}
              </pre>
            )}
          </CardContent>
        </Card>
      );
  }
};

// Main Dashboard Component
export default function Dashboard() {
  const [activeSection, setActiveSection] = useState(sectionOrder[0].key);
  const [data, setData] = useState(null);

  useEffect(() => {
    const storedData = localStorage.getItem('uploadReportResponse');
    if (storedData) {
      setData(JSON.parse(storedData));
    }
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <p className="text-lg text-gray-900">Loading data...</p>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'linear-gradient(to bottom right, #F8FAFC, #F1F5F9)',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Top Navigation */}
      <TopNav
        activeSection={activeSection}
        setActiveSection={setActiveSection}
      />

      {/* Header */}
      <div style={{ marginTop: '120px' }}>
        <Header companyName={data.company_name} />
      </div>

      {/* Main Content */}
      <main style={{
        flex: 1,
        padding: '2rem 0',
        background: 'linear-gradient(to bottom right, #F8FAFC, #F1F5F9)'
      }}>
        <div style={{
          maxWidth: '1280px',
          margin: '0 auto',
          padding: '0 1.5rem'
        }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(10px)',
            borderRadius: '1rem',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <SectionContent 
              activeSection={activeSection} 
              filteredData={data} 
              theme={{
                background: 'gray-50',
                text: 'gray-900',
                card: 'white',
                border: 'gray-200'
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
}