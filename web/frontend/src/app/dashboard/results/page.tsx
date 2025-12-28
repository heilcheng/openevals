"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  BarChart3,
  CheckCircle2,
  XCircle,
  Clock,
  Filter,
  ArrowRight,
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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { resultAPI, taskAPI, type TaskResultResponse, type TaskInfo } from "@/lib/api";

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-red-500" />;
    default:
      return <Clock className="h-4 w-4 text-blue-500" />;
  }
}

export default function ResultsPage() {
  const [results, setResults] = useState<TaskResultResponse[]>([]);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState({
    model_name: "",
    task_name: "",
  });

  useEffect(() => {
    async function fetchData() {
      try {
        const [resultsData, tasksData] = await Promise.all([
          resultAPI.list({ limit: 100 }),
          taskAPI.list(),
        ]);
        setResults(resultsData);
        setTasks(tasksData);
      } catch (error) {
        console.error("Failed to fetch results:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const filteredResults = results.filter((r) => {
    if (filters.model_name && !r.model_name.toLowerCase().includes(filters.model_name.toLowerCase())) {
      return false;
    }
    if (filters.task_name && r.task_name !== filters.task_name) {
      return false;
    }
    return true;
  });

  // Group results by model
  const resultsByModel = filteredResults.reduce((acc, result) => {
    if (!acc[result.model_name]) {
      acc[result.model_name] = [];
    }
    acc[result.model_name].push(result);
    return acc;
  }, {} as Record<string, TaskResultResponse[]>);

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Results</h1>
        <p className="text-muted-foreground">
          Browse and analyze benchmark results
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap gap-4 p-4">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Filters</span>
          </div>
          <Input
            placeholder="Search by model name..."
            value={filters.model_name}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, model_name: e.target.value }))
            }
            className="w-64"
          />
          <Select
            value={filters.task_name}
            onValueChange={(value) =>
              setFilters((prev) => ({
                ...prev,
                task_name: value === "all" ? "" : value,
              }))
            }
          >
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All tasks" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All tasks</SelectItem>
              {tasks.map((task) => (
                <SelectItem key={task.type} value={task.type}>
                  {task.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {(filters.model_name || filters.task_name) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFilters({ model_name: "", task_name: "" })}
            >
              Clear filters
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      {Object.keys(resultsByModel).length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BarChart3 className="h-12 w-12 text-muted-foreground" />
            <h3 className="mt-4 text-lg font-medium">No results found</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              {filters.model_name || filters.task_name
                ? "Try adjusting your filters"
                : "Complete some benchmarks to see results here"}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {Object.entries(resultsByModel).map(([modelName, modelResults], index) => (
            <motion.div
              key={modelName}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{modelName}</CardTitle>
                      <CardDescription>
                        {modelResults.length} task results
                      </CardDescription>
                    </div>
                    <Link href={`/dashboard/results/compare?models=${modelName}`}>
                      <Button variant="outline" size="sm" className="gap-2">
                        Compare
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </Link>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {modelResults.map((result) => (
                      <div
                        key={result.id}
                        className="flex flex-col rounded-lg border p-4"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{result.task_name}</span>
                          <StatusIcon status={result.status} />
                        </div>
                        {result.overall && (
                          <div className="mt-3 space-y-2">
                            {Object.entries(result.overall).map(([key, value]) => (
                              <div
                                key={key}
                                className="flex items-center justify-between text-sm"
                              >
                                <span className="text-muted-foreground capitalize">
                                  {key.replace(/_/g, " ")}
                                </span>
                                <span className="font-medium">
                                  {typeof value === "number"
                                    ? value > 1
                                      ? value.toFixed(1)
                                      : (value * 100).toFixed(1) + "%"
                                    : value}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                        {result.errors && result.errors.length > 0 && (
                          <div className="mt-3">
                            <Badge variant="destructive" className="text-xs">
                              {result.errors.length} error(s)
                            </Badge>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
