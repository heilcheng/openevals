"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import {
  BarChart3,
  CheckCircle2,
  XCircle,
  Clock,
  Filter,
  ArrowRight,
  Download,
  FileImage,
  FileText,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { resultAPI, taskAPI, type TaskResultResponse, type TaskInfo } from "@/lib/api";

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-4 w-4 text-muted-foreground" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-destructive" />;
    default:
      return <Clock className="h-4 w-4 text-muted-foreground" />;
  }
}

// Default tasks for when API fails
const defaultTasks: TaskInfo[] = [
  { type: "mmlu", name: "MMLU", description: "Multitask language understanding", metrics: ["accuracy"] },
  { type: "gsm8k", name: "GSM8K", description: "Grade school math", metrics: ["accuracy"] },
  { type: "truthfulqa", name: "TruthfulQA", description: "Truthfulness evaluation", metrics: ["mc1", "mc2"] },
  { type: "hellaswag", name: "HellaSwag", description: "Commonsense reasoning", metrics: ["accuracy"] },
  { type: "arc", name: "ARC", description: "Science reasoning", metrics: ["accuracy"] },
  { type: "humaneval", name: "HumanEval", description: "Code generation", metrics: ["pass@1"] },
];

export default function ResultsPage() {
  const [results, setResults] = useState<TaskResultResponse[]>([]);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState({
    model_name: "",
    task_name: "",
  });
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [resultsData, tasksData] = await Promise.all([
          resultAPI.list({ limit: 100 }),
          taskAPI.list().catch(() => defaultTasks),
        ]);
        setResults(resultsData);
        setTasks(tasksData.length > 0 ? tasksData : defaultTasks);
      } catch (error) {
        console.error("Failed to fetch results:", error);
        setTasks(defaultTasks);
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

  // Prepare chart data
  const chartData = Object.entries(resultsByModel).map(([model, modelResults]) => {
    const data: Record<string, string | number> = { model };
    modelResults.forEach((result) => {
      if (result.overall) {
        const mainMetric = Object.values(result.overall)[0];
        if (typeof mainMetric === "number") {
          data[result.task_name] = mainMetric > 1 ? mainMetric : mainMetric * 100;
        }
      }
    });
    return data;
  });

  const taskNames = [...new Set(filteredResults.map((r) => r.task_name))];

  // Download chart as PNG
  const downloadChart = async (format: "png" | "svg" | "csv") => {
    if (!chartRef.current) return;

    if (format === "csv") {
      // Export as CSV
      const headers = ["Model", ...taskNames];
      const rows = chartData.map((row) => {
        return [row.model, ...taskNames.map((task) => row[task] ?? "")].join(",");
      });
      const csv = [headers.join(","), ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `openevals-results-${new Date().toISOString().split("T")[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    // Export as PNG/SVG
    const svg = chartRef.current.querySelector("svg");
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);

    if (format === "svg") {
      const blob = new Blob([svgData], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `openevals-results-${new Date().toISOString().split("T")[0]}.svg`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    // PNG export
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width * 2;
      canvas.height = img.height * 2;
      if (ctx) {
        ctx.fillStyle = "#0a0a0a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.scale(2, 2);
        ctx.drawImage(img, 0, 0);
      }
      const pngUrl = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = `openevals-results-${new Date().toISOString().split("T")[0]}.png`;
      a.click();
    };

    img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
  };

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="loader" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Results</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Benchmark results and performance analysis
          </p>
        </div>
        {Object.keys(resultsByModel).length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="h-4 w-4" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => downloadChart("png")}>
                <FileImage className="h-4 w-4 mr-2" />
                Download PNG (for papers)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => downloadChart("svg")}>
                <FileImage className="h-4 w-4 mr-2" />
                Download SVG (vector)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => downloadChart("csv")}>
                <FileText className="h-4 w-4 mr-2" />
                Download CSV (data)
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap gap-4 p-4">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Filters</span>
          </div>
          <Input
            placeholder="Search by model..."
            value={filters.model_name}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, model_name: e.target.value }))
            }
            className="w-48"
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
            <SelectTrigger className="w-40">
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
              Clear
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Chart */}
      {chartData.length > 0 && taskNames.length > 0 && (
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Performance Comparison</CardTitle>
                <CardDescription className="text-xs">
                  Accuracy scores by model and task
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div ref={chartRef} className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="model"
                    tick={{ fill: "#888", fontSize: 12 }}
                    axisLine={{ stroke: "#444" }}
                  />
                  <YAxis
                    tick={{ fill: "#888", fontSize: 12 }}
                    axisLine={{ stroke: "#444" }}
                    domain={[0, 100]}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                      borderRadius: "6px",
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, ""]}
                  />
                  <Legend />
                  {taskNames.map((task, index) => (
                    <Bar
                      key={task}
                      dataKey={task}
                      fill={`hsl(${(index * 360) / taskNames.length}, 50%, 50%)`}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Table */}
      {Object.keys(resultsByModel).length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BarChart3 className="h-10 w-10 text-muted-foreground/50" />
            <p className="mt-4 text-sm text-muted-foreground">
              {filters.model_name || filters.task_name
                ? "No results match filters"
                : "Run benchmarks to see results"}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {Object.entries(resultsByModel).map(([modelName, modelResults]) => (
            <Card key={modelName}>
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">{modelName}</CardTitle>
                    <CardDescription className="text-xs">
                      {modelResults.length} task{modelResults.length !== 1 ? "s" : ""}
                    </CardDescription>
                  </div>
                  <Link href={`/dashboard/results/compare?models=${modelName}`}>
                    <Button variant="outline" size="sm" className="gap-1 text-xs">
                      Compare
                      <ArrowRight className="h-3 w-3" />
                    </Button>
                  </Link>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="border rounded-md">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left font-medium p-3">Task</th>
                        <th className="text-left font-medium p-3">Status</th>
                        <th className="text-left font-medium p-3">Metrics</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelResults.map((result, index) => (
                        <tr
                          key={result.id}
                          className={index !== modelResults.length - 1 ? "border-b" : ""}
                        >
                          <td className="p-3 font-medium">{result.task_name}</td>
                          <td className="p-3">
                            <StatusIcon status={result.status} />
                          </td>
                          <td className="p-3">
                            {result.overall ? (
                              <div className="flex flex-wrap gap-2">
                                {Object.entries(result.overall).map(([key, value]) => (
                                  <Badge key={key} variant="outline" className="font-mono text-xs">
                                    {key}: {typeof value === "number"
                                      ? value > 1
                                        ? value.toFixed(1)
                                        : (value * 100).toFixed(1) + "%"
                                      : value}
                                  </Badge>
                                ))}
                              </div>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
