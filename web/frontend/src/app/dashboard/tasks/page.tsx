"use client";

import { useEffect, useState } from "react";
import {
  ListChecks,
  Calculator,
  Brain,
  BookOpen,
  MessageSquare,
  Target,
  Lightbulb,
  CheckCircle2,
  Code,
  GraduationCap,
  Scale,
  Microscope,
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
import { taskAPI, type TaskInfo } from "@/lib/api";

const taskIcons: Record<string, React.ReactNode> = {
  gsm8k: <Calculator className="h-5 w-5" />,
  math: <Calculator className="h-5 w-5" />,
  truthfulqa: <CheckCircle2 className="h-5 w-5" />,
  mmlu: <Brain className="h-5 w-5" />,
  hellaswag: <Lightbulb className="h-5 w-5" />,
  arc: <Target className="h-5 w-5" />,
  winogrande: <MessageSquare className="h-5 w-5" />,
  humaneval: <Code className="h-5 w-5" />,
  gpqa: <GraduationCap className="h-5 w-5" />,
  ifeval: <Scale className="h-5 w-5" />,
  bbh: <Microscope className="h-5 w-5" />,
};

// Default tasks with full information
const defaultTasks: TaskInfo[] = [
  {
    type: "mmlu",
    name: "MMLU",
    description: "Massive Multitask Language Understanding - Tests knowledge across 57 subjects from STEM to humanities",
    metrics: ["accuracy"],
    default_config: { shot_count: 5 },
  },
  {
    type: "gsm8k",
    name: "GSM8K",
    description: "Grade School Math 8K - Mathematical reasoning benchmark testing arithmetic and word problem solving",
    metrics: ["accuracy", "exact_match"],
    default_config: { shot_count: 8 },
  },
  {
    type: "truthfulqa",
    name: "TruthfulQA",
    description: "Measures whether a language model generates truthful answers in adversarial settings",
    metrics: ["mc1", "mc2"],
    default_config: {},
  },
  {
    type: "hellaswag",
    name: "HellaSwag",
    description: "Commonsense reasoning about physical situations - predicting what happens next",
    metrics: ["accuracy"],
    default_config: {},
  },
  {
    type: "arc",
    name: "ARC",
    description: "AI2 Reasoning Challenge - Science questions requiring reasoning (Easy and Challenge splits)",
    metrics: ["accuracy"],
    default_config: { subset: "challenge" },
  },
  {
    type: "winogrande",
    name: "WinoGrande",
    description: "Large-scale Winograd Schema Challenge for commonsense reasoning",
    metrics: ["accuracy"],
    default_config: {},
  },
  {
    type: "humaneval",
    name: "HumanEval",
    description: "Code generation benchmark - Python function completion from docstrings",
    metrics: ["pass@1", "pass@10"],
    default_config: {},
  },
  {
    type: "gpqa",
    name: "GPQA",
    description: "Graduate-level Google-Proof Q&A - Expert-level questions in physics, biology, and chemistry",
    metrics: ["accuracy"],
    default_config: { subset: "diamond" },
  },
  {
    type: "math",
    name: "MATH",
    description: "Competition mathematics problems from AMC, AIME, and Olympiad levels",
    metrics: ["accuracy"],
    default_config: { shot_count: 4 },
  },
  {
    type: "ifeval",
    name: "IFEval",
    description: "Instruction Following Evaluation - Tests ability to follow specific formatting instructions",
    metrics: ["strict_accuracy", "loose_accuracy"],
    default_config: {},
  },
  {
    type: "bbh",
    name: "BBH",
    description: "BIG-Bench Hard - 23 challenging tasks from BIG-Bench requiring multi-step reasoning",
    metrics: ["accuracy"],
    default_config: { shot_count: 3 },
  },
];

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchTasks() {
      try {
        const data = await taskAPI.list().catch(() => []);

        // Merge API tasks with defaults, ensuring metrics exist
        const mergedTasks = defaultTasks.map((defaultTask) => {
          const apiTask = data.find((t) => t.type === defaultTask.type);
          return apiTask
            ? { ...defaultTask, ...apiTask, metrics: apiTask.metrics || defaultTask.metrics }
            : defaultTask;
        });

        // Add any API tasks not in defaults
        const additionalTasks = data.filter(
          (t) => !defaultTasks.find((d) => d.type === t.type)
        ).map((t) => ({ ...t, metrics: t.metrics || ["accuracy"] }));

        setTasks([...mergedTasks, ...additionalTasks]);
      } catch (error) {
        console.error("Failed to fetch tasks:", error);
        setTasks(defaultTasks);
      } finally {
        setIsLoading(false);
      }
    }

    fetchTasks();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Benchmark Tasks</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Available evaluation tasks for testing model capabilities
        </p>
      </div>

      {/* Tasks Grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-40 animate-pulse rounded-lg bg-muted/50"
            />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {tasks.map((task) => (
            <Card key={task.type} className="card-hover">
              <CardHeader className="pb-3">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-md border bg-background">
                    {taskIcons[task.type] || <ListChecks className="h-5 w-5" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <CardTitle className="text-base">{task.name}</CardTitle>
                    <p className="font-mono text-xs text-muted-foreground">
                      {task.type}
                    </p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2">
                  {task.description}
                </p>

                {/* Metrics */}
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {(task.metrics || []).map((metric) => (
                    <Badge
                      key={metric}
                      variant="secondary"
                      className="font-mono text-xs"
                    >
                      {metric}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Info Section */}
      <Card>
        <CardContent className="flex items-start gap-4 p-5">
          <div className="rounded-md border bg-background p-2">
            <BookOpen className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <h3 className="font-medium text-sm">About Benchmarks</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Each benchmark task evaluates different capabilities of language models.
              Select multiple tasks when creating a benchmark run for comprehensive evaluation.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
