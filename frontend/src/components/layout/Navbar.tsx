
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Search, User, Bell } from 'lucide-react';
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const Navbar = () => {
  const location = useLocation();
  
  const navItems = [
    { label: 'Home', to: '/' },
    { label: 'Market Insights', to: '/market-insights' },
    { label: 'Stock Screener', to: '/stock/:symbol' },
    { label: 'Compare Stocks', to: '/compare' },
    { label: 'Sector Analysis', to: '/sectors' },
    { label: 'Portfolio', to: '/portfolio' },
    { label: 'News', to: '/news' },
    // { label: 'Alerts', to: '/alerts' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-6">
          <Link to="/" className="items-center">
            <span className="text-xl font-bold text-primary">Quant</span>
            <span className="text-xl font-bold">AI</span>
          </Link>
          <nav className="hidden md:flex gap-6">
            {navItems.map((item) => (
              <Link
                key={item.label}
                to={item.to}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-primary",
                  location.pathname === item.to
                    ? "text-primary"
                    : "text-muted-foreground"
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-4">
          <div className="relative hidden md:block w-40 lg:w-64">
            {/* <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search stocks..."
              className="pl-8 rounded-full bg-muted"
            /> */}
          </div>
          {/* <Button size="icon" variant="ghost" className="text-muted-foreground">
            <Bell className="h-5 w-5" />
          </Button>
          <Button size="icon" variant="ghost" className="text-muted-foreground">
            <User className="h-5 w-5" />
          </Button>
          <Button size="sm" className="hidden md:inline-flex">
            Login
          </Button> */}
        </div>
      </div>
    </header>
  );
};

export default Navbar;
