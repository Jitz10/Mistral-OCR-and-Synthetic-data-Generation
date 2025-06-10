import React from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Plus } from 'lucide-react';
import { sectors } from '../data/portfolioData';
import { useToast } from "@/hooks/use-toast";

const addInvestmentSchema = z.object({
  ticker: z.string()
    .min(1, "Ticker is required")
    .max(5, "Ticker cannot exceed 5 characters")
    .toUpperCase(),
  name: z.string().min(1, "Company name is required"),
  shares: z.string()
    .min(1, "Number of shares is required")
    .transform((val) => Number(val))
    .refine((val) => val > 0, "Shares must be greater than 0"),
  avgCost: z.string()
    .min(1, "Average cost is required")
    .transform((val) => Number(val))
    .refine((val) => val > 0, "Average cost must be greater than 0"),
  sector: z.string().min(1, "Sector is required"),
});

type AddInvestmentFormData = z.infer<typeof addInvestmentSchema>;

interface AddInvestmentDialogProps {
  onAddInvestment: (investment: AddInvestmentFormData) => void;
}

export function AddInvestmentDialog({ onAddInvestment }: AddInvestmentDialogProps) {
  const { toast } = useToast();
  const form = useForm<AddInvestmentFormData>({
    resolver: zodResolver(addInvestmentSchema),
    defaultValues: {
      ticker: "",
      name: "",
      shares: "",
      avgCost: "",
      sector: "",
    },
  });

  const onSubmit = (data: AddInvestmentFormData) => {
    onAddInvestment(data);
    form.reset();
    toast({
      title: "Investment Added",
      description: `${data.ticker} has been added to your portfolio.`,
    });
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button size="sm">
          <Plus className="mr-2 h-4 w-4" />
          Add Investment
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Add New Investment</DialogTitle>
          <DialogDescription>
            Add a new stock to your portfolio. Enter the details below.
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="ticker"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Ticker Symbol</FormLabel>
                  <FormControl>
                    <Input placeholder="AAPL" {...field} />
                  </FormControl>
                  <FormDescription>
                    Enter the stock ticker symbol (e.g., AAPL for Apple)
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Company Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Apple Inc." {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="shares"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Number of Shares</FormLabel>
                  <FormControl>
                    <Input type="number" placeholder="100" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="avgCost"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Average Cost Per Share</FormLabel>
                  <FormControl>
                    <Input type="number" placeholder="150.00" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="sector"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Sector</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a sector" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {sectors.map((sector) => (
                        <SelectItem key={sector} value={sector}>
                          {sector}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="submit">Add Investment</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}