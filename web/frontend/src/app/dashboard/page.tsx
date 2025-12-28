"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
  Cpu,
  BarChart3,
  Trophy,
  Activity,
  TrendingUp,
  BookOpen,
  ExternalLink,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { benchmarkAPI, type BenchmarkRunResponse } from "@/lib/api";

interface StatsCard {
  title: string;
  value: number;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

function AnimatedCounter({ value }: { value: number }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (value === 0) {
      setDisplayValue(0);
      return;
    }

    const duration = 800;
    const steps = 40;
    const stepValue = value / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += stepValue;
      if (current >= value) {
        setDisplayValue(value);
        clearInterval(timer);
      } else {
        setDisplayValue(Math.floor(current));
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [value]);

  return <span>{displayValue}</span>;
}

function StatusBadge({ status }: { status: string }) {
  const variants: Record<
    string,
    { variant: "default" | "secondary" | "destructive" | "outline"; label: string }
  > = {
    pending: { variant: "secondary", label: "Pending" },
    running: { variant: "default", label: "Running" },
    completed: { variant: "outline", label: "Completed" },
    failed: { variant: "destructive", label: "Failed" },
    cancelled: { variant: "secondary", label: "Cancelled" },
  };

  const config = variants[status] || variants.pending;

  return (
    <Badge variant={config.variant} className="font-normal">
      {status === "running" && (
        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
      )}
      {config.label}
    </Badge>
  );
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<BenchmarkRunResponse[]>([]);
  const [stats, setStats] = useState<StatsCard[]>([
    {
      title: "Total Runs",
      value: 0,
      description: "All benchmark executions",
      icon: BarChart3,
    },
    {
      title: "Active",
      value: 0,
      description: "Currently running",
      icon: Activity,
    },
    {
      title: "Completed",
      value: 0,
      description: "Finished successfully",
      icon: CheckCircle2,
    },
    {
      title: "Models",
      value: 0,
      description: "Unique models tested",
      icon: Cpu,
    },
  ]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await benchmarkAPI.list({ limit: 5 });
        setRuns(data.items);

        const completed = data.items.filter(
          (r) => r.status === "completed"
        ).length;
        const running = data.items.filter((r) => r.status === "running").length;
        const models = new Set(
          data.items.flatMap((r) => (r.config?.models as string[]) || [])
        ).size;

        setStats([
          {
            title: "Total Runs",
            value: data.total,
            description: "All benchmark executions",
            icon: BarChart3,
          },
          {
            title: "Active",
            value: running,
            description: "Currently running",
            icon: Activity,
          },
          {
            title: "Completed",
            value: completed,
            description: "Finished successfully",
            icon: CheckCircle2,
          },
          {
            title: "Models",
            value: models,
            description: "Unique models tested",
            icon: Cpu,
          },
        ]);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Open-source LLM evaluation framework
          </p>
        </div>
        <Link href="/dashboard/benchmarks/new">
          <Button size="sm" className="gap-2">
            <Play className="h-4 w-4" />
            New Benchmark
          </Button>
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="card-hover">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold tracking-tight">
                <AnimatedCounter value={stat.value} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {stat.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid gap-4 md:grid-cols-3">
        {[
          {
            title: "Run Benchmark",
            description: "Evaluate models on standardized tasks",
            icon: Play,
            href: "/dashboard/benchmarks/new",
          },
          {
            title: "Compare Results",
            description: "Analyze performance across models",
            icon: TrendingUp,
            href: "/dashboard/results",
          },
          {
            title: "Leaderboard",
            description: "View aggregated rankings",
            icon: Trophy,
            href: "/dashboard/leaderboard",
          },
        ].map((action) => (
          <Link key={action.title} href={action.href}>
            <Card className="card-hover cursor-pointer h-full">
              <CardContent className="flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-md border bg-background">
                  <action.icon className="h-5 w-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-sm">{action.title}</h3>
                  <p className="text-xs text-muted-foreground truncate">
                    {action.description}
                  </p>
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Recent Runs */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between py-4">
          <div>
            <CardTitle className="text-base">Recent Runs</CardTitle>
            <CardDescription className="text-xs">
              Latest benchmark executions
            </CardDescription>
          </div>
          <Link href="/dashboard/benchmarks">
            <Button variant="ghost" size="sm" className="gap-1 text-xs">
              View all
              <ArrowRight className="h-3 w-3" />
            </Button>
          </Link>
        </CardHeader>
        <CardContent className="pt-0">
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-14 animate-pulse rounded-md bg-muted/50"
                />
              ))}
            </div>
          ) : runs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <BarChart3 className="h-8 w-8 text-muted-foreground/50" />
              <p className="mt-3 text-sm text-muted-foreground">
                No benchmark runs yet
              </p>
              <Link href="/dashboard/benchmarks/new" className="mt-3">
                <Button variant="outline" size="sm" className="gap-2">
                  <Play className="h-3 w-3" />
                  Create first benchmark
                </Button>
              </Link>
            </div>
          ) : (
            <div className="space-y-2">
              {runs.map((run) => (
                <Link key={run.id} href={`/dashboard/benchmarks/${run.id}`}>
                  <div className="flex items-center justify-between rounded-md border p-3 transition-colors hover:bg-accent">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted">
                        <BarChart3 className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <p className="font-medium text-sm truncate">
                          {run.name || `Run ${run.id.slice(0, 8)}`}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(run.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <StatusBadge status={run.status} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Resources for Researchers */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Resources</CardTitle>
          <CardDescription className="text-xs">
            Documentation and references
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { title: "Documentation", href: "/docs", icon: BookOpen },
              { title: "API Reference", href: "/api/docs", icon: ExternalLink },
              { title: "GitHub", href: "https://github.com/heilcheng/openevals", icon: ExternalLink },
              { title: "Paper", href: "#", icon: ExternalLink },
            ].map((link) => (
              <a
                key={link.title}
                href={link.href}
                target={link.href.startsWith("http") ? "_blank" : undefined}
                rel={link.href.startsWith("http") ? "noopener noreferrer" : undefined}
                className="flex items-center gap-2 rounded-md border p-3 text-sm hover:bg-accent transition-colors"
              >
                <link.icon className="h-4 w-4 text-muted-foreground" />
                {link.title}
              </a>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
