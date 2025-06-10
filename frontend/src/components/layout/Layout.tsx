
import React, { ReactNode } from 'react';
import Navbar from './Navbar';
import { MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import AiChatbot from '../shared/AiChatbot';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <div className="flex-1 flex flex-col">
        <main className="flex-1 overflow-auto">{children}</main>
        <div className="fixed bottom-6 right-6">
          {/* <Button size="lg" className="rounded-full h-14 w-14 bg-primary shadow-lg hover:shadow-xl">
            <MessageCircle className="h-6 w-6" />
          </Button> */}
          <AiChatbot />
        </div>
      </div>
    </div>
  );
};

export default Layout;
