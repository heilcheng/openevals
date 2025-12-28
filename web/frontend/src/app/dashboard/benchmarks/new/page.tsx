"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Play, X, Check, Cpu, ListChecks } from "lucide-react";
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

// Default model types
const defaultModelTypes: ModelTypeInfo[] = [
  { type: "gemma", name: "Gemma", sizes: ["2b", "7b", "9b", "27b"], default_size: "7b" },
  { type: "gemma3", name: "Gemma 3", sizes: ["1b", "4b", "12b", "27b"], default_size: "4b" },
  { type: "llama3", name: "Llama 3", sizes: ["8b", "70b"], default_size: "8b" },
  { type: "llama3.1", name: "Llama 3.1", sizes: ["8b", "70b", "405b"], default_size: "8b" },
  { type: "llama3.2", name: "Llama 3.2", sizes: ["1b", "3b", "11b", "90b"], default_size: "3b" },
  { type: "mistral", name: "Mistral", sizes: ["7b"], default_size: "7b" },
  { type: "mixtral", name: "Mixtral", sizes: ["8x7b", "8x22b"], default_size: "8x7b" },
  { type: "qwen2", name: "Qwen 2", sizes: ["0.5b", "1.5b", "7b", "72b"], default_size: "7b" },
  { type: "qwen2.5", name: "Qwen 2.5", sizes: ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"], default_size: "7b" },
  { type: "deepseek", name: "DeepSeek", sizes: ["7b", "67b"], default_size: "7b" },
  { type: "deepseek-r1", name: "DeepSeek-R1", sizes: ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"], default_size: "7b" },
  { type: "phi3", name: "Phi-3", sizes: ["mini", "small", "medium"], default_size: "mini" },
  { type: "olmo", name: "OLMo", sizes: ["1b", "7b"], default_size: "7b" },
  { type: "huggingface", name: "HuggingFace", sizes: ["custom"], default_size: "custom" },
];

// Default tasks
const defaultTasks: TaskInfo[] = [
  { type: "mmlu", name: "MMLU", description: "Multitask language understanding across 57 subjects", metrics: ["accuracy"] },
  { type: "gsm8k", name: "GSM8K", description: "Grade school math word problems", metrics: ["accuracy"] },
  { type: "truthfulqa", name: "TruthfulQA", description: "Truthfulness evaluation", metrics: ["mc1", "mc2"] },
  { type: "hellaswag", name: "HellaSwag", description: "Commonsense reasoning", metrics: ["accuracy"] },
  { type: "arc", name: "ARC", description: "Science reasoning questions", metrics: ["accuracy"] },
  { type: "winogrande", name: "WinoGrande", description: "Commonsense reasoning", metrics: ["accuracy"] },
  { type: "humaneval", name: "HumanEval", description: "Python code generation", metrics: ["pass@1"] },
  { type: "gpqa", name: "GPQA", description: "Graduate-level science Q&A", metrics: ["accuracy"] },
  { type: "math", name: "MATH", description: "Competition mathematics", metrics: ["accuracy"] },
  { type: "ifeval", name: "IFEval", description: "Instruction following", metrics: ["accuracy"] },
  { type: "bbh", name: "BBH", description: "BIG-Bench Hard reasoning tasks", metrics: ["accuracy"] },
];

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
        const tasksData = await taskAPI.list().catch(() => []);

        // Merge API tasks with defaults, ensuring metrics exist
        const mergedTasks = defaultTasks.map((defaultTask) => {
          const apiTask = tasksData.find((t) => t.type === defaultTask.type);
          return apiTask
            ? { ...defaultTask, ...apiTask, metrics: apiTask.metrics || defaultTask.metrics }
            : defaultTask;
        });

        // Add any API tasks not in defaults
        const additionalTasks = tasksData.filter(
          (t) => !defaultTasks.find((d) => d.type === t.type)
        );

        setModelTypes(defaultModelTypes);
        setTasks([...mergedTasks, ...additionalTasks]);
      } catch (error) {
        console.error("Failed to fetch data:", error);
        setModelTypes(defaultModelTypes);
        setTasks(defaultTasks);
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
        <div className="loader" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/dashboard/benchmarks">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-xl font-semibold">New Benchmark</h1>
          <p className="text-sm text-muted-foreground">
            Configure and run a benchmark evaluation
          </p>
        </div>
      </div>

      {/* Form */}
      <div className="space-y-6">
        {/* Basic Info */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Basic Information</CardTitle>
            <CardDescription className="text-xs">
              Optional name and description for this run
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="My Benchmark Run"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1.5"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Description</label>
              <Input
                placeholder="Evaluating model performance on reasoning tasks..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="mt-1.5"
              />
            </div>
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              <CardTitle className="text-base">Select Models</CardTitle>
            </div>
            <CardDescription className="text-xs">
              {selectedModels.length} model{selectedModels.length !== 1 ? "s" : ""} selected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {modelTypes.map((model) => (
                <button
                  key={model.type}
                  onClick={() => toggleModel(model.type)}
                  className={`relative flex flex-col items-start rounded-md border p-3 text-left transition-colors ${
                    selectedModels.includes(model.type)
                      ? "border-foreground bg-accent"
                      : "hover:bg-accent/50"
                  }`}
                >
                  {selectedModels.includes(model.type) && (
                    <div className="absolute right-2 top-2 rounded-full bg-foreground p-0.5">
                      <Check className="h-2.5 w-2.5 text-background" />
                    </div>
                  )}
                  <span className="font-medium text-sm">{model.name}</span>
                  <span className="mt-0.5 text-xs text-muted-foreground">
                    {model.sizes.join(", ")}
                  </span>
                </button>
              ))}
            </div>

            {selectedModels.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {selectedModels.map((modelId) => (
                  <Badge key={modelId} variant="secondary" className="gap-1 text-xs">
                    {modelTypes.find((m) => m.type === modelId)?.name || modelId}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleModel(modelId);
                      }}
                      className="ml-0.5 hover:text-destructive"
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
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2">
              <ListChecks className="h-4 w-4" />
              <CardTitle className="text-base">Select Tasks</CardTitle>
            </div>
            <CardDescription className="text-xs">
              {selectedTasks.length} task{selectedTasks.length !== 1 ? "s" : ""} selected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2">
              {tasks.map((task) => (
                <button
                  key={task.type}
                  onClick={() => toggleTask(task.type)}
                  className={`relative flex flex-col items-start rounded-md border p-3 text-left transition-colors ${
                    selectedTasks.includes(task.type)
                      ? "border-foreground bg-accent"
                      : "hover:bg-accent/50"
                  }`}
                >
                  {selectedTasks.includes(task.type) && (
                    <div className="absolute right-2 top-2 rounded-full bg-foreground p-0.5">
                      <Check className="h-2.5 w-2.5 text-background" />
                    </div>
                  )}
                  <span className="font-medium text-sm">{task.name}</span>
                  <span className="mt-0.5 text-xs text-muted-foreground line-clamp-1">
                    {task.description}
                  </span>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {(task.metrics || []).map((metric) => (
                      <Badge key={metric} variant="outline" className="text-xs py-0">
                        {metric}
                      </Badge>
                    ))}
                  </div>
                </button>
              ))}
            </div>

            {selectedTasks.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {selectedTasks.map((taskType) => (
                  <Badge key={taskType} variant="secondary" className="gap-1 text-xs">
                    {tasks.find((t) => t.type === taskType)?.name || taskType}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleTask(taskType);
                      }}
                      className="ml-0.5 hover:text-destructive"
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
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Submit */}
        <div className="flex justify-end gap-3">
          <Link href="/dashboard/benchmarks">
            <Button variant="outline" size="sm">Cancel</Button>
          </Link>
          <Button
            onClick={handleSubmit}
            disabled={
              isSubmitting ||
              selectedModels.length === 0 ||
              selectedTasks.length === 0
            }
            size="sm"
            className="gap-2"
          >
            {isSubmitting ? (
              <>
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Creating...
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                Start Benchmark
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
