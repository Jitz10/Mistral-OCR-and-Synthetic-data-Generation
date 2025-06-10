import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, BotIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

const AiChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I'm QuantAI Assistant. How can I help with your financial research today?",
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  // Scroll to the bottom when new messages are added
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!inputValue.trim()) return;

    const newUserMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages([...messages, newUserMessage]);
    setInputValue('');
    setIsLoading(true);

    console.log('Initiating API call with query:', inputValue); // Debug: Confirm API call start

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        console.log('API call timed out'); // Debug: Timeout triggered
      }, 10000);

      const response = await fetch('http://crew.maazmalik2004.space/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add any specific headers required by the API (e.g., Authorization) if needed
        },
        body: JSON.stringify({
          query: inputValue,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      console.log('API response status:', response.status); // Debug: Log status

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const data = await response.json();
      console.log('API response data:', data); // Debug: Log full response

      // Adjust based on expected API response structure
      const aiResponse = data.response || data.answer || data.message || data.result || 'No valid response received from the server.';

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: aiResponse,
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error: any) {
      console.error('API Error:', error.name, error.message, error.stack); // Detailed error logging
      let errorContent = 'Sorry, something went wrong. Please try again later.';

      if (error.name === 'AbortError') {
        errorContent = 'The request timed out. Please try again.';
      } else if (error.message.includes('Failed to fetch')) {
        errorContent = 'Unable to connect to the server. Check your network or try again.';
      } else if (error.message.includes('status 403')) {
        errorContent = 'Access denied. The server refused the request.';
      } else if (error.message.includes('status 500')) {
        errorContent = 'Server error. Please try again later.';
      }

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: errorContent,
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      toast({
        title: 'Error',
        description: errorContent,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
      console.log('API call completed'); // Debug: Confirm completion
    }
  };

  return (
    <>
      <Button
        size="lg"
        className="fixed bottom-6 right-6 rounded-full h-14 w-14 bg-primary shadow-lg hover:shadow-xl z-50"
        onClick={toggleChatbot}
      >
        {isOpen ? <X className="h-6 w-6" /> : <MessageCircle className="h-6 w-6" />}
      </Button>

      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-80 md:w-96 shadow-xl z-40 border border-border animate-fade-in">
          <CardHeader className="bg-primary text-primary-foreground">
            <CardTitle className="text-sm flex items-center gap-2">
              <BotIcon size={18} />
              QuantAI Assistant
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    'flex',
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  )}
                >
                  <div
                    className={cn(
                      'rounded-lg px-4 py-2 max-w-[85%] text-sm',
                      message.sender === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'
                    )}
                  >
                    {message.content}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="max-w-[85%] rounded-lg px-4 py-2 text-sm bg-muted">
                    <div className="flex space-x-1">
                      <div
                        className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                        style={{ animationDelay: '0ms' }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                        style={{ animationDelay: '150ms' }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
                        style={{ animationDelay: '300ms' }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </CardContent>
          <CardFooter className="border-t p-2">
            <form onSubmit={handleSendMessage} className="flex w-full gap-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about stocks, markets, portfolios..."
                className="flex-1"
                disabled={isLoading}
              />
              <Button type="submit" size="icon" disabled={isLoading || !inputValue.trim()}>
                <Send size={18} />
              </Button>
            </form>
          </CardFooter>
        </Card>
      )}
    </>
  );
};

export default AiChatbot;