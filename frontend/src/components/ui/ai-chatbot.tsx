import { useState, useRef, useEffect } from "react";
import { MessageSquare, X, ChevronUp, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

export function AIChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "Hi, I'm QuantAI assistant. How can I help with your financial research today?",
      role: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    console.log("Initiating API call with query:", inputValue); // Debug: Confirm API call start

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        console.log("API call timed out"); // Debug: Timeout triggered
      }, 10000);

      const response = await fetch("http://crew.maazmalik2004.space/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Add any required headers (e.g., Authorization) if needed in Postman
        },
        body: JSON.stringify({
          query: inputValue,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      console.log("API response status:", response.status); // Debug: Log status

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const data = await response.json();
      console.log("API response data:", data); // Debug: Log full response

      // Adjust based on actual API response structure (based on Postman output)
      const aiResponse = data.response || data.answer || data.message || data.result || "No valid response received.";

      const assistantMessage: Message = {
        id: Date.now().toString(),
        content: aiResponse,
        role: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error("API Error:", error.name, error.message, error.stack); // Detailed error logging
      let errorContent = "Sorry, something went wrong. Please try again later.";

      if (error.name === "AbortError") {
        errorContent = "The request timed out. Please try again.";
      } else if (error.message.includes("Failed to fetch")) {
        errorContent = "Unable to connect to the server. Check your network or try again.";
      } else if (error.message.includes("status 403")) {
        errorContent = "Access denied. The server refused the request.";
      } else if (error.message.includes("status 500")) {
        errorContent = "Server error. Please try again later.";
      }

      const errorMessage: Message = {
        id: Date.now().toString(),
        content: errorContent,
        role: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      toast({
        title: "Error",
        description: errorContent,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      console.log("API call completed"); // Debug: Confirm completion
    }

    scrollToBottom();
  };

  return (
    <>
      {/* Floating button */}
      <div className="fixed bottom-4 right-4 z-50">
        <Button
          onClick={toggleChatbot}
          className={cn(
            "w-12 h-12 rounded-full shadow-lg bg-blue-600 hover:bg-blue-700 text-white",
            isOpen ? "rotate-90" : "animate-pulse-subtle"
          )}
        >
          {isOpen ? <X size={22} /> : <MessageSquare size={22} />}
        </Button>
      </div>

      {/* Chatbot window */}
      <div
        className={cn(
          "fixed bottom-20 right-4 z-50 w-80 md:w-96 bg-white dark:bg-slate-900 rounded-lg shadow-xl transition-all duration-300 transform",
          isOpen
            ? "scale-100 opacity-100 translate-y-0"
            : "scale-95 opacity-0 translate-y-8 pointer-events-none"
        )}
      >
        <Card className="border overflow-hidden h-[450px] flex flex-col">
          {/* Header */}
          <div className="bg-quantai-purple text-white p-3 flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <MessageSquare size={18} />
              <h3 className="font-semibold text-primary">QuantAI Assistant</h3>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleChatbot}
              className="h-8 w-8 text-blue hover:bg-primary"
            >
              <ChevronUp size={18} />
            </Button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex",
                  msg.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg px-4 py-2 text-sm",
                    msg.role === "user"
                      ? "bg-primary text-white"
                      : "bg-gray-100 dark:bg-slate-800 text-foreground"
                  )}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-lg px-4 py-2 text-sm bg-gray-100 dark:bg-slate-800 text-foreground">
                  <div className="flex space-x-1">
                    <div
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="border-t p-3 flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about stocks, markets, or data..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              size="icon"
              className="bg-primary hover:bg-primary"
            >
              <Send size={18} />
            </Button>
          </form>
        </Card>
      </div>
    </>
  );
}