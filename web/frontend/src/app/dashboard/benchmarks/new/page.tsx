"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowLeft, Play, Plus, X, Check, Cpu, ListChecks } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  benchmarkAPI,
  modelAPI,
  taskAPI,
  type ModelTypeInfo,
  type TaskInfo,
} from "@/lib/api";

export default function NewBenchmarkPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [modelTypes, setModelTypes] = useState<ModelTypeInfo[]>([]);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [modelsData, tasksData] = await Promise.all([
          modelAPI.types(),
          taskAPI.list(),
        ]);
        setModelTypes(modelsData);
        setTasks(tasksData);
      } catch (error) {
        console.error("Failed to fetch data:", error);
        setError("Failed to load configuration options");
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const toggleTask = (taskType: string) => {
    setSelectedTasks((prev) =>
      prev.includes(taskType)
        ? prev.filter((t) => t !== taskType)
        : [...prev, taskType]
    );
  };

  const handleSubmit = async () => {
    if (selectedModels.length === 0 || selectedTasks.length === 0) {
      setError("Please select at least one model and one task");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await benchmarkAPI.create({
        name: name || `Benchmark ${new Date().toLocaleDateString()}`,
        description,
        models: selectedModels,
        tasks: selectedTasks,
      });

      router.push(`/dashboard/benchmarks/${result.id}`);
    } catch (error) {
      console.error("Failed to create benchmark:", error);
      setError(
        error instanceof Error ? error.message : "Failed to create benchmark"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/dashboard/benchmarks">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">Create New Benchmark</h1>
          <p className="text-muted-foreground">
            Configure and run a benchmark evaluation
          </p>
        </div>
      </div>

      {/* Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Basic Info */}
        <Card>
          <CardHeader>
            <CardTitle>Basic Information</CardTitle>
            <CardDescription>
              Give your benchmark a name and description
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Name (optional)</label>
              <Input
                placeholder="My Benchmark Run"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">
                Description (optional)
              </label>
              <Input
                placeholder="Evaluating model performance on reasoning tasks..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="mt-1"
              />
            </div>
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary" />
              <CardTitle>Select Models</CardTitle>
            </div>
            <CardDescription>
              Choose which models to evaluate ({selectedModels.length} selected)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {modelTypes.map((model) => (
                <motion.button
                  key={model.type}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => toggleModel(model.type)}
                  className={`relative flex flex-col items-start rounded-lg border p-4 text-left transition-colors ${
                    selectedModels.includes(model.type)
                      ? "border-primary bg-primary/5"
                      : "hover:bg-accent"
                  }`}
                >
                  {selectedModels.includes(model.type) && (
                    <div className="absolute right-2 top-2 rounded-full bg-primary p-1">
                      <Check className="h-3 w-3 text-white" />
                    </div>
                  )}
                  <span className="font-medium">{model.name}</span>
                  <span className="mt-1 text-sm text-muted-foreground">
                    Sizes: {model.sizes.join(", ")}
                  </span>
                </motion.button>
              ))}
            </div>

            {/* Selected Models */}
            {selectedModels.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {selectedModels.map((modelId) => (
                  <Badge key={modelId} variant="secondary" className="gap-1">
                    {modelTypes.find((m) => m.type === modelId)?.name || modelId}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleModel(modelId);
                      }}
                      className="ml-1 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Task Selection */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <ListChecks className="h-5 w-5 text-primary" />
              <CardTitle>Select Tasks</CardTitle>
            </div>
            <CardDescription>
              Choose which benchmark tasks to run ({selectedTasks.length}{" "}
              selected)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2">
              {tasks.map((task) => (
                <motion.button
                  key={task.type}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => toggleTask(task.type)}
                  className={`relative flex flex-col items-start rounded-lg border p-4 text-left transition-colors ${
                    selectedTasks.includes(task.type)
                      ? "border-primary bg-primary/5"
                      : "hover:bg-accent"
                  }`}
                >
                  {selectedTasks.includes(task.type) && (
                    <div className="absolute right-2 top-2 rounded-full bg-primary p-1">
                      <Check className="h-3 w-3 text-white" />
                    </div>
                  )}
                  <span className="font-medium">{task.name}</span>
                  <span className="mt-1 text-sm text-muted-foreground line-clamp-2">
                    {task.description}
                  </span>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {task.metrics.map((metric) => (
                      <Badge key={metric} variant="outline" className="text-xs">
                        {metric}
                      </Badge>
                    ))}
                  </div>
                </motion.button>
              ))}
            </div>

            {/* Selected Tasks */}
            {selectedTasks.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {selectedTasks.map((taskType) => (
                  <Badge key={taskType} variant="secondary" className="gap-1">
                    {tasks.find((t) => t.type === taskType)?.name || taskType}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleTask(taskType);
                      }}
                      className="ml-1 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive"
          >
            {error}
          </motion.div>
        )}

        {/* Submit */}
        <div className="flex justify-end gap-4">
          <Link href="/dashboard/benchmarks">
            <Button variant="outline">Cancel</Button>
          </Link>
          <Button
            onClick={handleSubmit}
            disabled={
              isSubmitting ||
              selectedModels.length === 0 ||
              selectedTasks.length === 0
            }
            className="gap-2"
          >
            {isSubmitting ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Creating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Start Benchmark
              </>
            )}
          </Button>
        </div>
      </motion.div>
    </div>
  );
}
