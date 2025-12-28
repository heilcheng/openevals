"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div className="lg:pl-56 pl-16">
        <Header />
        <main className="p-6 max-w-7xl">{children}</main>
      </div>
    </div>
  );
}
