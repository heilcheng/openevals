"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Plus,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  MoreVertical,
  Trash2,
  Eye,
  StopCircle,
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

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { variant: "default" | "secondary" | "destructive" | "success" | "warning"; icon: React.ReactNode }> = {
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
      icon: <StopCircle className="mr-1 h-3 w-3" />,
    },
  };

  const c = config[status] || config.pending;

  return (
    <Badge variant={c.variant} className="capitalize">
      {c.icon}
      {status}
    </Badge>
  );
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<BenchmarkRunResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    async function fetchRuns() {
      try {
        const data = await benchmarkAPI.list({ limit: 50 });
        setRuns(data.items);
      } catch (error) {
        console.error("Failed to fetch benchmarks:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchRuns();
  }, []);

  const filteredRuns = runs.filter((run) => {
    if (filter === "all") return true;
    return run.status === filter;
  });

  const statusCounts = {
    all: runs.length,
    running: runs.filter((r) => r.status === "running").length,
    completed: runs.filter((r) => r.status === "completed").length,
    failed: runs.filter((r) => r.status === "failed").length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Benchmarks</h1>
          <p className="text-muted-foreground">
            Manage and monitor your benchmark runs
          </p>
        </div>
        <Link href="/dashboard/benchmarks/new">
          <Button className="gap-2 gradient-primary text-primary-foreground">
            <Plus className="h-4 w-4" />
            New Benchmark
          </Button>
        </Link>
      </div>

      {/* Status Filters */}
      <div className="flex gap-2">
        {(["all", "running", "completed", "failed"] as const).map((status) => (
          <Button
            key={status}
            variant={filter === status ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter(status)}
            className="capitalize"
          >
            {status}
            <span className="ml-2 rounded-full bg-background/20 px-2 py-0.5 text-xs">
              {statusCounts[status]}
            </span>
          </Button>
        ))}
      </div>

      {/* Benchmarks List */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-40 animate-pulse rounded-lg bg-muted/50"
            />
          ))}
        </div>
      ) : filteredRuns.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <div className="rounded-full bg-muted p-6">
              <Play className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="mt-6 text-lg font-semibold">No benchmarks found</h3>
            <p className="mt-2 text-center text-muted-foreground">
              {filter !== "all"
                ? `No ${filter} benchmarks at the moment`
                : "Get started by creating your first benchmark"}
            </p>
            {filter === "all" && (
              <Link href="/dashboard/benchmarks/new" className="mt-6">
                <Button>
                  <Plus className="mr-2 h-4 w-4" />
                  Create Benchmark
                </Button>
              </Link>
            )}
          </CardContent>
        </Card>
      ) : (
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid gap-4 md:grid-cols-2 lg:grid-cols-3"
        >
          {filteredRuns.map((run) => {
            const models = (run.config?.models as string[]) || [];
            const tasks = (run.config?.tasks as string[]) || [];

            return (
              <motion.div key={run.id} variants={item}>
                <Link href={`/dashboard/benchmarks/${run.id}`}>
                  <Card className="spotlight-card group cursor-pointer transition-all hover:shadow-lg hover:shadow-primary/5">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <CardTitle className="text-base font-semibold truncate group-hover:text-primary transition-colors">
                            {run.name || `Benchmark ${run.id.slice(0, 8)}`}
                          </CardTitle>
                          <CardDescription className="mt-1 text-xs">
                            {new Date(run.created_at).toLocaleDateString()} at{" "}
                            {new Date(run.created_at).toLocaleTimeString()}
                          </CardDescription>
                        </div>
                        <StatusBadge status={run.status} />
                      </div>
                    </CardHeader>
                    <CardContent>
                      {/* Progress for running benchmarks */}
                      {run.status === "running" && (
                        <div className="mb-4">
                          <div className="mb-1 flex justify-between text-xs">
                            <span className="text-muted-foreground">
                              {run.current_task
                                ? `Running: ${run.current_task}`
                                : "In progress..."}
                            </span>
                            <span className="font-medium">
                              {run.progress_percent || 0}%
                            </span>
                          </div>
                          <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                            <motion.div
                              className="h-full gradient-primary"
                              initial={{ width: 0 }}
                              animate={{ width: `${run.progress_percent || 0}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Models & Tasks */}
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1.5">
                            Models ({models.length})
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {models.slice(0, 3).map((model) => (
                              <Badge
                                key={model}
                                variant="secondary"
                                className="text-xs"
                              >
                                {model}
                              </Badge>
                            ))}
                            {models.length > 3 && (
                              <Badge variant="outline" className="text-xs">
                                +{models.length - 3}
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-1.5">
                            Tasks ({tasks.length})
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {tasks.slice(0, 3).map((task) => (
                              <Badge
                                key={task}
                                variant="outline"
                                className="text-xs"
                              >
                                {task}
                              </Badge>
                            ))}
                            {tasks.length > 3 && (
                              <Badge variant="outline" className="text-xs">
                                +{tasks.length - 3}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              </motion.div>
            );
          })}
        </motion.div>
      )}
    </div>
  );
}
