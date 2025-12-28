"use client";

import {
  RadarChart as RechartsRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

interface RadarChartProps {
  data: Array<{
    subject: string;
    [key: string]: string | number;
  }>;
  dataKeys: string[];
  colors?: string[];
}

const defaultColors = [
  "hsl(172, 66%, 50%)",
  "hsl(280, 100%, 70%)",
  "hsl(43, 96%, 56%)",
  "hsl(330, 100%, 65%)",
  "hsl(200, 100%, 60%)",
];

export function ModelRadarChart({
  data,
  dataKeys,
  colors = defaultColors,
}: RadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RechartsRadar cx="50%" cy="50%" outerRadius="80%" data={data}>
        <PolarGrid stroke="hsl(var(--border))" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[0, 1]}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        {dataKeys.map((key, index) => (
          <Radar
            key={key}
            name={key}
            dataKey={key}
            stroke={colors[index % colors.length]}
            fill={colors[index % colors.length]}
            fillOpacity={0.2}
            strokeWidth={2}
            animationDuration={1000}
          />
        ))}
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "8px",
          }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, ""]}
        />
        <Legend />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}

export function ComparisonRadar({
  modelScores,
}: {
  modelScores: Record<string, Record<string, number>>;
}) {
  const models = Object.keys(modelScores);
  const tasks = models.length > 0 ? Object.keys(modelScores[models[0]]) : [];

  const data = tasks.map((task) => ({
    subject: task,
    ...models.reduce(
      (acc, model) => ({
        ...acc,
        [model]: modelScores[model][task] || 0,
      }),
      {}
    ),
  }));

  return <ModelRadarChart data={data} dataKeys={models} />;
}
