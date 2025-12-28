import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenEvals - LLM Benchmark Suite",
  description: "Open-source evaluation framework for large language models",
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
