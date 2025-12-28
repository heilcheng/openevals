"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
  Cpu,
  BarChart3,
  Trophy,
  Zap,
  Sparkles,
  TrendingUp,
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
import { ShimmerButton } from "@/components/magic/shimmer-button";
import { AnimatedGradientText } from "@/components/magic/animated-gradient";
import { benchmarkAPI, type BenchmarkRunResponse } from "@/lib/api";

interface StatsCard {
  title: string;
  value: number;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  gradient: string;
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

function AnimatedCounter({ value }: { value: number }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (value === 0) {
      setDisplayValue(0);
      return;
    }

    const duration = 1000;
    const steps = 60;
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
    { variant: "default" | "secondary" | "destructive" | "success" | "warning"; icon: React.ReactNode }
  > = {
    pending: {
      variant: "secondary",
      icon: <Clock className="mr-1 h-3 w-3" />,
    },
    running: {
      variant: "warning",
      icon: <Play className="mr-1 h-3 w-3 animate-pulse" />,
    },
    completed: {
      variant: "success",
      icon: <CheckCircle2 className="mr-1 h-3 w-3" />,
    },
    failed: {
      variant: "destructive",
      icon: <XCircle className="mr-1 h-3 w-3" />,
    },
    cancelled: {
      variant: "secondary",
      icon: <XCircle className="mr-1 h-3 w-3" />,
    },
  };

  const config = variants[status] || variants.pending;

  return (
    <Badge variant={config.variant} className="capitalize">
      {config.icon}
      {status}
    </Badge>
  );
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<BenchmarkRunResponse[]>([]);
  const [stats, setStats] = useState<StatsCard[]>([
    {
      title: "Total Benchmarks",
      value: 0,
      description: "All benchmark runs",
      icon: BarChart3,
      gradient: "from-cyan-500 to-teal-500",
    },
    {
      title: "Running",
      value: 0,
      description: "Currently executing",
      icon: Zap,
      gradient: "from-amber-500 to-orange-500",
    },
    {
      title: "Completed",
      value: 0,
      description: "Successfully finished",
      icon: CheckCircle2,
      gradient: "from-emerald-500 to-green-500",
    },
    {
      title: "Models Tested",
      value: 0,
      description: "Unique models evaluated",
      icon: Cpu,
      gradient: "from-violet-500 to-purple-500",
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
            title: "Total Benchmarks",
            value: data.total,
            description: "All benchmark runs",
            icon: BarChart3,
            gradient: "from-cyan-500 to-teal-500",
          },
          {
            title: "Running",
            value: running,
            description: "Currently executing",
            icon: Zap,
            gradient: "from-amber-500 to-orange-500",
          },
          {
            title: "Completed",
            value: completed,
            description: "Successfully finished",
            icon: CheckCircle2,
            gradient: "from-emerald-500 to-green-500",
          },
          {
            title: "Models Tested",
            value: models,
            description: "Unique models evaluated",
            icon: Cpu,
            gradient: "from-violet-500 to-purple-500",
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
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-background via-background to-primary/5 p-8"
      >
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
          <div className="absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-accent/10 blur-3xl" />
        </div>

        <div className="relative">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Sparkles className="h-4 w-4 text-primary" />
            <span>AI Model Benchmarking Platform</span>
          </div>
          <h1 className="mt-2 text-4xl font-bold tracking-tight">
            Welcome to{" "}
            <AnimatedGradientText>Gemma Benchmark</AnimatedGradientText>
          </h1>
          <p className="mt-3 max-w-2xl text-lg text-muted-foreground">
            Evaluate and compare language models across multiple benchmarks.
            Track performance, visualize results, and build comprehensive leaderboards.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link href="/dashboard/benchmarks/new">
              <ShimmerButton className="gap-2">
                <Play className="h-4 w-4" />
                New Benchmark
              </ShimmerButton>
            </Link>
            <Link href="/dashboard/leaderboard">
              <Button variant="outline" className="gap-2 border-primary/50 hover:bg-primary/10">
                <Trophy className="h-4 w-4" />
                View Leaderboard
              </Button>
            </Link>
          </div>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {stats.map((stat, index) => (
          <motion.div key={stat.title} variants={item}>
            <Card className="group relative overflow-hidden transition-all hover:shadow-lg hover:shadow-primary/5">
              {/* Gradient overlay on hover */}
              <div className={`absolute inset-0 bg-gradient-to-br ${stat.gradient} opacity-0 transition-opacity group-hover:opacity-5`} />

              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.title}
                </CardTitle>
                <motion.div
                  className={`rounded-xl bg-gradient-to-br ${stat.gradient} p-2.5 shadow-lg`}
                  whileHover={{ scale: 1.1, rotate: 5 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <stat.icon className="h-4 w-4 text-white" />
                </motion.div>
              </CardHeader>
              <CardContent>
                <div className="text-4xl font-bold tracking-tight">
                  <AnimatedCounter value={stat.value} />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {stat.description}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid gap-4 md:grid-cols-3"
      >
        {[
          {
            title: "Run Benchmark",
            description: "Start a new model evaluation",
            icon: Play,
            href: "/dashboard/benchmarks/new",
            gradient: "from-primary to-accent",
          },
          {
            title: "Compare Models",
            description: "Analyze performance differences",
            icon: TrendingUp,
            href: "/dashboard/results",
            gradient: "from-emerald-500 to-teal-500",
          },
          {
            title: "Leaderboard",
            description: "View top performing models",
            icon: Trophy,
            href: "/dashboard/leaderboard",
            gradient: "from-amber-500 to-orange-500",
          },
        ].map((action, index) => (
          <motion.div
            key={action.title}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 + index * 0.1 }}
          >
            <Link href={action.href}>
              <Card className="group cursor-pointer transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/50">
                <CardContent className="flex items-center gap-4 p-6">
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${action.gradient} shadow-lg transition-transform group-hover:scale-110`}
                  >
                    <action.icon className="h-6 w-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold group-hover:text-primary transition-colors">
                      {action.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {action.description}
                    </p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground opacity-0 transition-all group-hover:opacity-100 group-hover:translate-x-1" />
                </CardContent>
              </Card>
            </Link>
          </motion.div>
        ))}
      </motion.div>

      {/* Recent Runs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Recent Benchmark Runs</CardTitle>
              <CardDescription>
                Your latest benchmark executions
              </CardDescription>
            </div>
            <Link href="/dashboard/benchmarks">
              <Button variant="ghost" className="gap-2 group">
                View All
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="h-16 animate-pulse rounded-lg bg-muted/50"
                  />
                ))}
              </div>
            ) : runs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="rounded-full bg-muted p-4">
                  <BarChart3 className="h-8 w-8 text-muted-foreground" />
                </div>
                <h3 className="mt-4 text-lg font-medium">No benchmarks yet</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Start your first benchmark to see results here
                </p>
                <Link href="/dashboard/benchmarks/new" className="mt-4">
                  <Button className="gap-2">
                    <Play className="h-4 w-4" />
                    Create Benchmark
                  </Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {runs.map((run, index) => (
                  <motion.div
                    key={run.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Link href={`/dashboard/benchmarks/${run.id}`}>
                      <div className="group flex items-center justify-between rounded-lg border p-4 transition-all hover:bg-accent/50 hover:border-primary/50">
                        <div className="flex items-center gap-4">
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-accent/20">
                            <BarChart3 className="h-5 w-5 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium group-hover:text-primary transition-colors">
                              {run.name || `Benchmark ${run.id.slice(0, 8)}`}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {new Date(run.created_at).toLocaleDateString()} at{" "}
                              {new Date(run.created_at).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <StatusBadge status={run.status} />
                          <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 transition-all group-hover:opacity-100 group-hover:translate-x-1" />
                        </div>
                      </div>
                    </Link>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
