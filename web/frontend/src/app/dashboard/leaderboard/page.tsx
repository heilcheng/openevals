"use client";

import { useEffect, useState, useRef } from "react";
import { Trophy, Medal, Award, TrendingUp, Download, FileImage, FileText } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { resultAPI, type LeaderboardEntry } from "@/lib/api";

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) {
    return (
      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground text-background">
        <Trophy className="h-3.5 w-3.5" />
      </div>
    );
  }
  if (rank === 2) {
    return (
      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-muted-foreground/50 text-background">
        <Medal className="h-3.5 w-3.5" />
      </div>
    );
  }
  if (rank === 3) {
    return (
      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-muted-foreground/30 text-foreground">
        <Award className="h-3.5 w-3.5" />
      </div>
    );
  }
  return (
    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-muted text-sm font-medium">
      {rank}
    </div>
  );
}

export default function LeaderboardPage() {
  const [leaderboard, setLeaderboard] = useState<{
    entries: LeaderboardEntry[];
    tasks: string[];
    total_models: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTask, setSelectedTask] = useState<string>("all");
  const [chartType, setChartType] = useState<"bar" | "radar">("bar");
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        const data = await resultAPI.leaderboard();
        setLeaderboard(data);
      } catch (error) {
        console.error("Failed to fetch leaderboard:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchLeaderboard();
  }, []);

  const getSortedEntries = () => {
    if (!leaderboard) return [];

    if (selectedTask === "all") {
      return [...leaderboard.entries].sort(
        (a, b) => b.average_score - a.average_score
      );
    }

    return [...leaderboard.entries]
      .filter((e) => selectedTask in e.task_scores)
      .sort(
        (a, b) =>
          (b.task_scores[selectedTask] || 0) -
          (a.task_scores[selectedTask] || 0)
      );
  };

  const getScore = (entry: LeaderboardEntry) => {
    if (selectedTask === "all") {
      return entry.average_score;
    }
    return entry.task_scores[selectedTask] || 0;
  };

  const sortedEntries = getSortedEntries();

  // Prepare chart data
  const barChartData = sortedEntries.slice(0, 10).map((entry) => ({
    model: entry.model_name.length > 15
      ? entry.model_name.substring(0, 15) + "..."
      : entry.model_name,
    score: getScore(entry) * 100,
  }));

  const radarChartData = leaderboard?.tasks.map((task) => {
    const data: Record<string, string | number> = { task };
    sortedEntries.slice(0, 5).forEach((entry) => {
      data[entry.model_name] = (entry.task_scores[task] || 0) * 100;
    });
    return data;
  }) || [];

  // Download chart
  const downloadChart = async (format: "png" | "svg" | "csv") => {
    if (!chartRef.current) return;

    if (format === "csv") {
      const headers = ["Rank", "Model", "Score", ...(leaderboard?.tasks || [])];
      const rows = sortedEntries.map((entry, index) => {
        const taskScores = (leaderboard?.tasks || []).map(
          (task) => ((entry.task_scores[task] || 0) * 100).toFixed(1)
        );
        return [
          index + 1,
          entry.model_name,
          (getScore(entry) * 100).toFixed(1),
          ...taskScores,
        ].join(",");
      });
      const csv = [headers.join(","), ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `openevals-leaderboard-${new Date().toISOString().split("T")[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    const svg = chartRef.current.querySelector("svg");
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);

    if (format === "svg") {
      const blob = new Blob([svgData], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `openevals-leaderboard-${new Date().toISOString().split("T")[0]}.svg`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

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
      a.download = `openevals-leaderboard-${new Date().toISOString().split("T")[0]}.png`;
      a.click();
    };

    img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
  };

  // Generate LaTeX table
  const downloadLatex = () => {
    const tasks = leaderboard?.tasks || [];
    const header = `\\begin{table}[h]
\\centering
\\caption{Model Performance Comparison}
\\label{tab:results}
\\begin{tabular}{l${"c".repeat(tasks.length + 1)}}
\\toprule
Model & ${tasks.map((t) => t.toUpperCase()).join(" & ")} & Avg \\\\
\\midrule`;

    const rows = sortedEntries.map((entry) => {
      const scores = tasks.map((task) =>
        ((entry.task_scores[task] || 0) * 100).toFixed(1)
      );
      const avg = (getScore(entry) * 100).toFixed(1);
      return `${entry.model_name} & ${scores.join(" & ")} & ${avg} \\\\`;
    });

    const footer = `\\bottomrule
\\end{tabular}
\\end{table}`;

    const latex = [header, ...rows, footer].join("\n");
    const blob = new Blob([latex], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `openevals-leaderboard-${new Date().toISOString().split("T")[0]}.tex`;
    a.click();
    URL.revokeObjectURL(url);
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
          <h1 className="text-2xl font-semibold tracking-tight">Leaderboard</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Model rankings by benchmark performance
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <TrendingUp className="h-3.5 w-3.5" />
            {leaderboard?.total_models || 0} models
          </div>
          {sortedEntries.length > 0 && (
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
                  Download PNG
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => downloadChart("svg")}>
                  <FileImage className="h-4 w-4 mr-2" />
                  Download SVG
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => downloadChart("csv")}>
                  <FileText className="h-4 w-4 mr-2" />
                  Download CSV
                </DropdownMenuItem>
                <DropdownMenuItem onClick={downloadLatex}>
                  <FileText className="h-4 w-4 mr-2" />
                  Download LaTeX Table
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      {/* Task Filter */}
      {leaderboard && leaderboard.tasks.length > 0 && (
        <Tabs value={selectedTask} onValueChange={setSelectedTask}>
          <TabsList className="h-8">
            <TabsTrigger value="all" className="text-xs px-3">Overall</TabsTrigger>
            {leaderboard.tasks.map((task) => (
              <TabsTrigger key={task} value={task} className="text-xs px-3 uppercase">
                {task}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      )}

      {/* Charts */}
      {sortedEntries.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {/* Bar Chart */}
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="text-base">Score Distribution</CardTitle>
              <CardDescription className="text-xs">
                Top 10 models by {selectedTask === "all" ? "average" : selectedTask} score
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div ref={chartRef} className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barChartData} layout="vertical" margin={{ left: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: "#888", fontSize: 11 }}
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <YAxis
                      type="category"
                      dataKey="model"
                      tick={{ fill: "#888", fontSize: 11 }}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "4px",
                        fontSize: "12px",
                      }}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, "Score"]}
                    />
                    <Bar dataKey="score" fill="#888" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Radar Chart */}
          {leaderboard && leaderboard.tasks.length >= 3 && sortedEntries.length >= 2 && (
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-base">Task Comparison</CardTitle>
                <CardDescription className="text-xs">
                  Performance across tasks (top 5 models)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarChartData}>
                      <PolarGrid stroke="#333" />
                      <PolarAngleAxis
                        dataKey="task"
                        tick={{ fill: "#888", fontSize: 10 }}
                      />
                      <PolarRadiusAxis
                        tick={{ fill: "#666", fontSize: 9 }}
                        domain={[0, 100]}
                      />
                      {sortedEntries.slice(0, 5).map((entry, index) => (
                        <Radar
                          key={entry.model_name}
                          name={entry.model_name}
                          dataKey={entry.model_name}
                          stroke={`hsl(${index * 72}, 50%, 50%)`}
                          fill={`hsl(${index * 72}, 50%, 50%)`}
                          fillOpacity={0.1}
                        />
                      ))}
                      <Legend wrapperStyle={{ fontSize: "11px" }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Full Leaderboard Table */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Rankings</CardTitle>
          <CardDescription className="text-xs">
            {selectedTask === "all"
              ? "Average performance across all tasks"
              : `Performance on ${selectedTask}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sortedEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <Trophy className="h-10 w-10 text-muted-foreground/50" />
              <p className="mt-4 text-sm text-muted-foreground">
                Run benchmarks to see rankings
              </p>
            </div>
          ) : (
            <div className="border rounded-md">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left font-medium p-3 w-12">#</th>
                    <th className="text-left font-medium p-3">Model</th>
                    <th className="text-left font-medium p-3">Score</th>
                    <th className="text-left font-medium p-3 hidden md:table-cell">Tasks</th>
                    <th className="text-left font-medium p-3 hidden lg:table-cell">Runs</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedEntries.map((entry, index) => {
                    const score = getScore(entry);
                    return (
                      <tr
                        key={entry.model_name}
                        className={index !== sortedEntries.length - 1 ? "border-b" : ""}
                      >
                        <td className="p-3">
                          <RankBadge rank={index + 1} />
                        </td>
                        <td className="p-3 font-medium">{entry.model_name}</td>
                        <td className="p-3">
                          <span className="font-mono">{(score * 100).toFixed(1)}%</span>
                        </td>
                        <td className="p-3 hidden md:table-cell">
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(entry.task_scores)
                              .slice(0, 4)
                              .map(([task, taskScore]) => (
                                <Badge
                                  key={task}
                                  variant={task === selectedTask ? "default" : "outline"}
                                  className="text-xs font-mono"
                                >
                                  {task}: {(taskScore * 100).toFixed(0)}%
                                </Badge>
                              ))}
                            {Object.keys(entry.task_scores).length > 4 && (
                              <Badge variant="outline" className="text-xs">
                                +{Object.keys(entry.task_scores).length - 4}
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="p-3 hidden lg:table-cell text-muted-foreground">
                          {entry.runs_count || entry.total_runs || "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
