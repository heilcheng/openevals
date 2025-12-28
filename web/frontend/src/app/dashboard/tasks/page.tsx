"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ListChecks,
  Calculator,
  Brain,
  BookOpen,
  MessageSquare,
  Target,
  Lightbulb,
  CheckCircle2,
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
  gsm8k: <Calculator className="h-6 w-6" />,
  truthfulqa: <CheckCircle2 className="h-6 w-6" />,
  mmlu: <Brain className="h-6 w-6" />,
  hellaswag: <Lightbulb className="h-6 w-6" />,
  arc: <Target className="h-6 w-6" />,
  winogrande: <MessageSquare className="h-6 w-6" />,
};

const taskColors: Record<string, string> = {
  gsm8k: "from-blue-500 to-indigo-600",
  truthfulqa: "from-green-500 to-emerald-600",
  mmlu: "from-purple-500 to-violet-600",
  hellaswag: "from-amber-500 to-orange-600",
  arc: "from-rose-500 to-pink-600",
  winogrande: "from-cyan-500 to-teal-600",
};

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12 },
  },
};

const item = {
  hidden: { opacity: 0, y: 30, scale: 0.9 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { type: "spring", stiffness: 300, damping: 24 },
  },
};

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchTasks() {
      try {
        const data = await taskAPI.list();
        setTasks(data);
      } catch (error) {
        console.error("Failed to fetch tasks:", error);
        // Mock data for development
        setTasks([
          {
            type: "gsm8k",
            name: "GSM8K",
            description: "Grade School Math 8K - Mathematical reasoning benchmark testing arithmetic and word problem solving abilities",
            metrics: ["accuracy", "exact_match"],
          },
          {
            type: "truthfulqa",
            name: "TruthfulQA",
            description: "Measures whether a language model generates truthful answers to questions in adversarial settings",
            metrics: ["accuracy", "mc1", "mc2"],
          },
          {
            type: "mmlu",
            name: "MMLU",
            description: "Massive Multitask Language Understanding - Tests knowledge across 57 subjects from STEM to humanities",
            metrics: ["accuracy", "per_subject"],
          },
          {
            type: "hellaswag",
            name: "HellaSwag",
            description: "Commonsense reasoning about situations - can the model predict what happens next?",
            metrics: ["accuracy"],
          },
        ]);
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
        <h1 className="text-2xl font-bold tracking-tight">Benchmark Tasks</h1>
        <p className="text-muted-foreground">
          Available evaluation tasks for testing model capabilities
        </p>
      </div>

      {/* Tasks Grid */}
      {isLoading ? (
        <div className="grid gap-6 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="h-48 animate-pulse rounded-xl bg-muted/50"
            />
          ))}
        </div>
      ) : (
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid gap-6 md:grid-cols-2"
        >
          {tasks.map((task) => (
            <motion.div key={task.type} variants={item}>
              <Card className="group relative overflow-hidden transition-all hover:shadow-lg hover:shadow-primary/5">
                {/* Gradient background on hover */}
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${
                    taskColors[task.type] || "from-gray-500 to-gray-600"
                  } opacity-0 transition-opacity duration-500 group-hover:opacity-[0.03]`}
                />

                {/* Animated border */}
                <div className="absolute inset-0 rounded-lg opacity-0 transition-opacity group-hover:opacity-100">
                  <div
                    className={`absolute inset-0 rounded-lg bg-gradient-to-r ${
                      taskColors[task.type] || "from-gray-500 to-gray-600"
                    } opacity-20 blur-sm`}
                  />
                </div>

                <CardHeader className="relative pb-3">
                  <div className="flex items-start gap-4">
                    <div
                      className={`flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br ${
                        taskColors[task.type] || "from-gray-500 to-gray-600"
                      } text-white shadow-lg transition-transform group-hover:scale-110`}
                    >
                      {taskIcons[task.type] || <ListChecks className="h-6 w-6" />}
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-xl transition-colors group-hover:text-primary">
                        {task.name}
                      </CardTitle>
                      <p className="mt-0.5 font-mono text-xs text-muted-foreground">
                        {task.type}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {task.description}
                  </p>

                  {/* Metrics */}
                  <div className="mt-4">
                    <p className="text-xs font-medium text-muted-foreground mb-2">
                      Evaluation Metrics
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {task.metrics.map((metric) => (
                        <Badge
                          key={metric}
                          variant="secondary"
                          className="font-mono text-xs transition-colors group-hover:bg-primary/10"
                        >
                          {metric}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Info Section */}
      <Card className="bg-muted/30">
        <CardContent className="flex items-center gap-4 p-6">
          <div className="rounded-full bg-primary/10 p-3">
            <BookOpen className="h-6 w-6 text-primary" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold">About Benchmarks</h3>
            <p className="text-sm text-muted-foreground">
              Each benchmark task evaluates different capabilities of language models.
              Select multiple tasks when creating a benchmark run to get a comprehensive
              evaluation of your model&apos;s performance.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
