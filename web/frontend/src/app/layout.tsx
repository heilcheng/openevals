import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Benchmark Suite",
  description: "Systematic evaluation framework for large language models",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-background antialiased">
        {children}
      </body>
    </html>
  );
}
