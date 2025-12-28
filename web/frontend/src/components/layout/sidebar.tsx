"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Play,
  Trophy,
  BarChart3,
  Settings,
  Cpu,
  ListChecks,
  FileText,
} from "lucide-react";

import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const navItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Benchmarks",
    href: "/dashboard/benchmarks",
    icon: Play,
  },
  {
    title: "Models",
    href: "/dashboard/models",
    icon: Cpu,
  },
  {
    title: "Tasks",
    href: "/dashboard/tasks",
    icon: ListChecks,
  },
  {
    title: "Results",
    href: "/dashboard/results",
    icon: BarChart3,
  },
  {
    title: "Leaderboard",
    href: "/dashboard/leaderboard",
    icon: Trophy,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <TooltipProvider delayDuration={0}>
      <aside className="fixed left-0 top-0 z-40 h-screen w-16 border-r bg-background lg:w-56">
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex h-14 items-center border-b px-4">
            <Link href="/dashboard" className="flex items-center gap-2">
              <FileText className="h-5 w-5 lg:hidden" />
              <span className="hidden text-sm font-semibold tracking-tight lg:block">
                OpenEvals
              </span>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-0.5 p-2">
            {navItems.map((item) => {
              const isActive =
                pathname === item.href ||
                (item.href !== "/dashboard" && pathname.startsWith(item.href));

              return (
                <Tooltip key={item.href}>
                  <TooltipTrigger asChild>
                    <Link
                      href={item.href}
                      className={cn(
                        "flex items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                        isActive && "bg-accent text-foreground font-medium"
                      )}
                    >
                      <item.icon className="h-4 w-4" />
                      <span className="hidden lg:block">{item.title}</span>
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent side="right" className="lg:hidden">
                    {item.title}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t p-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  href="/dashboard/settings"
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                    pathname === "/dashboard/settings" && "bg-accent text-foreground font-medium"
                  )}
                >
                  <Settings className="h-4 w-4" />
                  <span className="hidden lg:block">Settings</span>
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right" className="lg:hidden">
                Settings
              </TooltipContent>
            </Tooltip>
            <div className="hidden lg:block px-3 py-2 mt-2">
              <p className="text-xs text-muted-foreground">
                v1.0.0
              </p>
            </div>
          </div>
        </div>
      </aside>
    </TooltipProvider>
  );
}
