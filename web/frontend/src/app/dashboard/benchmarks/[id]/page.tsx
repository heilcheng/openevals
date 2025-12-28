"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  StopCircle,
  RefreshCw,
  Cpu,
  ListChecks,
} from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { benchmarkAPI, resultAPI, type BenchmarkRunResponse, type TaskResultResponse } from "@/lib/api";
import { useWebSocket } from "@/hooks/use-websocket";

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "running":
      return <Play className="h-5 w-5 text-amber-500" />;
    case "completed":
      return <CheckCircle2 className="h-5 w-5 text-green-500" />;
    case "failed":
      return <XCircle className="h-5 w-5 text-red-500" />;
    case "cancelled":
      return <StopCircle className="h-5 w-5 text-gray-500" />;
    default:
      return <Clock className="h-5 w-5 text-blue-500" />;
  }
}

function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, "default" | "secondary" | "destructive" | "success" | "warning"> = {
    pending: "secondary",
    running: "warning",
    completed: "success",
    failed: "destructive",
    cancelled: "secondary",
  };

  return (
    <Badge variant={variants[status] || "secondary"} className="capitalize">
      {status}
    </Badge>
  );
}

export default function BenchmarkDetailPage() {
  const params = useParams();
  const router = useRouter();
  const runId = params.id as string;

  const [run, setRun] = useState<BenchmarkRunResponse | null>(null);
  const [results, setResults] = useState<TaskResultResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCancelling, setIsCancelling] = useState(false);

  const handleWebSocketMessage = useCallback((message: Record<string, unknown>) => {
    if (message.type === "progress") {
      setRun((prev) =>
        prev
          ? {
              ...prev,
              progress_percent: (message.progress as number) || prev.progress_percent,
              current_model: (message.current_model as string) || prev.current_model,
              current_task: (message.current_task as string) || prev.current_task,
              status: (message.status as string) || prev.status,
            }
          : null
      );
    } else if (message.type === "completed" || message.type === "failed") {
      // Refresh full data
      fetchData();
    }
  }, []);

  const { isConnected } = useWebSocket(
    run?.status === "running" ? runId : null,
    { onMessage: handleWebSocketMessage }
  );

  const fetchData = async () => {
    try {
      const [runData, resultsData] = await Promise.all([
        benchmarkAPI.get(runId),
        resultAPI.list({ run_id: runId }),
      ]);
      setRun(runData);
      setResults(resultsData);
    } catch (error) {
      console.error("Failed to fetch benchmark:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    // Poll for updates if running and not connected to WebSocket
    let interval: NodeJS.Timeout | null = null;
    if (run?.status === "running" && !isConnected) {
      interval = setInterval(fetchData, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [runId, run?.status, isConnected]);

  const handleCancel = async () => {
    if (!run) return;
    setIsCancelling(true);
    try {
      await benchmarkAPI.cancel(runId);
      setRun((prev) => (prev ? { ...prev, status: "cancelled" } : null));
    } catch (error) {
      console.error("Failed to cancel benchmark:", error);
    } finally {
      setIsCancelling(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!run) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <h2 className="text-lg font-medium">Benchmark not found</h2>
        <Link href="/dashboard/benchmarks" className="mt-4">
          <Button>Back to Benchmarks</Button>
        </Link>
      </div>
    );
  }

  const models = (run.config?.models as string[]) || [];
  const tasks = (run.config?.tasks as string[]) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/benchmarks">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <div>
            <div className="flex items-center gap-3">
              <StatusIcon status={run.status} />
              <h1 className="text-2xl font-bold">
                {run.name || `Benchmark ${run.id.slice(0, 8)}`}
              </h1>
              <StatusBadge status={run.status} />
            </div>
            {run.description && (
              <p className="mt-1 text-muted-foreground">{run.description}</p>
            )}
          </div>
        </div>

        <div className="flex gap-2">
          {run.status === "running" && (
            <Button
              variant="destructive"
              onClick={handleCancel}
              disabled={isCancelling}
            >
              {isCancelling ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <StopCircle className="mr-2 h-4 w-4" />
              )}
              Cancel
            </Button>
          )}
          <Button variant="outline" onClick={fetchData}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Progress Section */}
      <AnimatePresence>
        {run.status === "running" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card className="border-primary/50 bg-primary/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="h-3 w-3 animate-pulse rounded-full bg-primary" />
                  Running Benchmark
                  {isConnected && (
                    <Badge variant="outline" className="ml-2 text-xs">
                      Live
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  {run.current_model && run.current_task
                    ? `Evaluating ${run.current_model} on ${run.current_task}`
                    : "Starting benchmark..."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span className="font-medium">{run.progress_percent}%</span>
                  </div>
                  <Progress value={run.progress_percent} className="h-3" />
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Message */}
      {run.error_message && (
        <Card className="border-destructive/50 bg-destructive/10">
          <CardContent className="p-4">
            <p className="text-destructive">{run.error_message}</p>
          </CardContent>
        </Card>
      )}

      {/* Configuration */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary" />
              <CardTitle>Models</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {models.map((model) => (
                <Badge key={model} variant="secondary">
                  {model}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <ListChecks className="h-5 w-5 text-primary" />
              <CardTitle>Tasks</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {tasks.map((task) => (
                <Badge key={task} variant="secondary">
                  {task}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>
              Task results for this benchmark run
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {results.map((result, index) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div className="flex items-center gap-4">
                      <StatusIcon status={result.status} />
                      <div>
                        <p className="font-medium">{result.model_name}</p>
                        <p className="text-sm text-muted-foreground">
                          {result.task_name}
                        </p>
                      </div>
                    </div>
                    {result.overall && (
                      <div className="text-right">
                        {Object.entries(result.overall).map(([key, value]) => (
                          <div key={key} className="text-sm">
                            <span className="text-muted-foreground">{key}: </span>
                            <span className="font-medium">
                              {typeof value === "number"
                                ? (value * 100).toFixed(1) + "%"
                                : value}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  {index < results.length - 1 && <Separator className="my-4" />}
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Details</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-sm text-muted-foreground">ID</dt>
              <dd className="font-mono text-sm">{run.id}</dd>
            </div>
            <div>
              <dt className="text-sm text-muted-foreground">Created</dt>
              <dd className="text-sm">
                {new Date(run.created_at).toLocaleString()}
              </dd>
            </div>
            {run.started_at && (
              <div>
                <dt className="text-sm text-muted-foreground">Started</dt>
                <dd className="text-sm">
                  {new Date(run.started_at).toLocaleString()}
                </dd>
              </div>
            )}
            {run.completed_at && (
              <div>
                <dt className="text-sm text-muted-foreground">Completed</dt>
                <dd className="text-sm">
                  {new Date(run.completed_at).toLocaleString()}
                </dd>
              </div>
            )}
          </dl>
        </CardContent>
      </Card>
    </div>
  );
}
