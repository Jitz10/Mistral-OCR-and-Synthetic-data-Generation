import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { 
  Home, 
  PieChart, 
  LineChart, 
  BarChart3, 
  Briefcase, 
  Newspaper, 
  Bell, 
  Settings, 
  ChevronLeft, 
  ChevronRight, 
  Search,
  ArrowRightLeft,
} from 'lucide-react';

interface SidebarItemProps {
  icon: React.ElementType;
  label: string;
  to: string;
  isActive?: boolean;
  isCollapsed: boolean;
}

const SidebarItem = ({ icon: Icon, label, to, isActive = false, isCollapsed }: SidebarItemProps) => {
  return (
    <Link
      to={to}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all duration-200",
        isActive 
          ? "bg-sidebar-primary text-sidebar-primary-foreground" 
          : "text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
      )}
    >
      <Icon size={20} />
      {!isCollapsed && <span>{label}</span>}
    </Link>
  );
};

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { icon: Home, label: 'Dashboard', to: '/' },
    { icon: PieChart, label: 'Market Insights', to: '/market-insights' },
    { icon: Search, label: 'Stock Screener', to: '/screener' },
    { icon: ArrowRightLeft, label: 'Compare Stocks', to: '/compare' },
    { icon: LineChart, label: 'Sector Analysis', to: '/sectors' },
    { icon: Briefcase, label: 'My Portfolio', to: '/portfolio' },
    { icon: Newspaper, label: 'News & Forecasts', to: '/news' },
    { icon: Bell, label: 'Alerts', to: '/alerts' },
    { icon: Settings, label: 'Settings', to: '/settings' }
  ];

  return (
    // <aside
    //   className={cn(
    //     "bg-sidebar flex flex-col h-screen sticky top-0 border-r border-sidebar-border transition-all duration-300",
    //     isCollapsed ? "w-16" : "w-64"
    //   )}
    // >
    //   <div className="flex h-14 items-center justify-between border-b border-sidebar-border px-4">
    //     {!isCollapsed && (
    //       <h2 className="text-xl font-bold text-sidebar-foreground">
    //         <span className="text-primary">Quant</span><span className="text-sidebar-foreground">AI</span>
    //       </h2>
    //     )}
    //     <button
    //       onClick={() => setIsCollapsed(!isCollapsed)}
    //       className="ml-auto rounded-lg p-1 text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
    //     >
    //       {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
    //     </button>
    //   </div>

    //   <nav className="flex-1 space-y-1 px-2 py-4">
    //     {navItems.map((item) => (
    //       <SidebarItem
    //         key={item.label}
    //         icon={item.icon}
    //         label={item.label}
    //         to={item.to}
    //         isActive={item.to === '/'}
    //         isCollapsed={isCollapsed}
    //       />
    //     ))}
    //   </nav>

      {/* <div className="border-t border-sidebar-border p-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-sidebar-primary flex items-center justify-center text-sidebar-primary-foreground">
            <span className="text-sm font-medium">JD</span>
          </div>
          {!isCollapsed && (
            <div>
              <p className="text-sm font-medium text-sidebar-foreground">John Doe</p>
              <p className="text-xs text-sidebar-foreground/70">Premium Plan</p>
            </div>
          )}
        </div>
      </div> */}
    // </aside>
  );
};

export default Sidebar;