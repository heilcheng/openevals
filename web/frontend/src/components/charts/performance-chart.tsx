"use client";

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";

interface PerformanceChartProps {
  data: Array<{
    name: string;
    score: number;
    [key: string]: string | number;
  }>;
  dataKey?: string;
  colors?: string[];
  showGrid?: boolean;
  animated?: boolean;
}

const defaultColors = [
  "hsl(172, 66%, 50%)",
  "hsl(280, 100%, 70%)",
  "hsl(43, 96%, 56%)",
  "hsl(330, 100%, 65%)",
  "hsl(200, 100%, 60%)",
];

export function PerformanceBarChart({
  data,
  dataKey = "score",
  colors = defaultColors,
  showGrid = true,
  animated = true,
}: PerformanceChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
        {showGrid && (
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            opacity={0.5}
          />
        )}
        <XAxis
          dataKey="name"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          axisLine={{ stroke: "hsl(var(--border))" }}
        />
        <YAxis
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          axisLine={{ stroke: "hsl(var(--border))" }}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
          }}
          labelStyle={{ color: "hsl(var(--foreground))" }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Score"]}
        />
        <Bar
          dataKey={dataKey}
          radius={[4, 4, 0, 0]}
          animationDuration={animated ? 1000 : 0}
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

interface HeatmapData {
  models: string[];
  tasks: string[];
  scores: Record<string, Record<string, number>>;
}

export function PerformanceHeatmap({ data }: { data: HeatmapData }) {
  const { models, tasks, scores } = data;

  const getColor = (score: number) => {
    // Color gradient from red (0) to yellow (0.5) to green (1)
    if (score < 0.5) {
      const t = score / 0.5;
      return `hsl(${0 + t * 43}, 70%, 50%)`;
    } else {
      const t = (score - 0.5) / 0.5;
      return `hsl(${43 + t * 117}, 70%, 50%)`;
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="border border-border bg-muted/50 p-3 text-left text-sm font-medium">
              Model / Task
            </th>
            {tasks.map((task) => (
              <th
                key={task}
                className="border border-border bg-muted/50 p-3 text-center text-sm font-medium"
              >
                {task}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((model, modelIndex) => (
            <motion.tr
              key={model}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: modelIndex * 0.1 }}
            >
              <td className="border border-border p-3 text-sm font-medium">
                {model}
              </td>
              {tasks.map((task, taskIndex) => {
                const score = scores[model]?.[task] ?? 0;
                return (
                  <motion.td
                    key={task}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{
                      delay: modelIndex * 0.1 + taskIndex * 0.05,
                    }}
                    className="border border-border p-3 text-center"
                    style={{ backgroundColor: getColor(score) }}
                  >
                    <span className="font-mono text-sm font-bold text-white drop-shadow-md">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </motion.td>
                );
              })}
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface MultiBarData {
  name: string;
  [key: string]: string | number;
}

export function MultiModelBarChart({
  data,
  models,
  colors = defaultColors,
}: {
  data: MultiBarData[];
  models: string[];
  colors?: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="hsl(var(--border))"
          opacity={0.5}
        />
        <XAxis
          dataKey="name"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          axisLine={{ stroke: "hsl(var(--border))" }}
        />
        <YAxis
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          axisLine={{ stroke: "hsl(var(--border))" }}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "8px",
          }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, ""]}
        />
        <Legend />
        {models.map((model, index) => (
          <Bar
            key={model}
            dataKey={model}
            fill={colors[index % colors.length]}
            radius={[4, 4, 0, 0]}
            animationDuration={1000}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
