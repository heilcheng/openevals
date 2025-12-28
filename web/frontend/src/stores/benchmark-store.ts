import { create } from "zustand";
import type { BenchmarkRunResponse, TaskResultResponse } from "@/lib/api";

interface BenchmarkState {
  // Active benchmark run being viewed/tracked
  activeBenchmark: BenchmarkRunResponse | null;
  setActiveBenchmark: (benchmark: BenchmarkRunResponse | null) => void;

  // Progress updates from WebSocket
  progress: {
    percent: number;
    currentModel: string | null;
    currentTask: string | null;
    status: string;
    message: string;
  };
  updateProgress: (progress: Partial<BenchmarkState["progress"]>) => void;

  // Results cache
  results: Record<string, TaskResultResponse[]>;
  setResults: (runId: string, results: TaskResultResponse[]) => void;
  addResult: (runId: string, result: TaskResultResponse) => void;

  // UI state
  isCreating: boolean;
  setIsCreating: (isCreating: boolean) => void;

  // Selected items for new benchmark
  selectedModels: string[];
  toggleModel: (model: string) => void;
  setSelectedModels: (models: string[]) => void;

  selectedTasks: string[];
  toggleTask: (task: string) => void;
  setSelectedTasks: (tasks: string[]) => void;

  // Reset selection
  resetSelection: () => void;
}

export const useBenchmarkStore = create<BenchmarkState>((set) => ({
  // Active benchmark
  activeBenchmark: null,
  setActiveBenchmark: (benchmark) => set({ activeBenchmark: benchmark }),

  // Progress
  progress: {
    percent: 0,
    currentModel: null,
    currentTask: null,
    status: "pending",
    message: "",
  },
  updateProgress: (progress) =>
    set((state) => ({
      progress: { ...state.progress, ...progress },
    })),

  // Results
  results: {},
  setResults: (runId, results) =>
    set((state) => ({
      results: { ...state.results, [runId]: results },
    })),
  addResult: (runId, result) =>
    set((state) => ({
      results: {
        ...state.results,
        [runId]: [...(state.results[runId] || []), result],
      },
    })),

  // UI state
  isCreating: false,
  setIsCreating: (isCreating) => set({ isCreating }),

  // Selected models
  selectedModels: [],
  toggleModel: (model) =>
    set((state) => ({
      selectedModels: state.selectedModels.includes(model)
        ? state.selectedModels.filter((m) => m !== model)
        : [...state.selectedModels, model],
    })),
  setSelectedModels: (models) => set({ selectedModels: models }),

  // Selected tasks
  selectedTasks: [],
  toggleTask: (task) =>
    set((state) => ({
      selectedTasks: state.selectedTasks.includes(task)
        ? state.selectedTasks.filter((t) => t !== task)
        : [...state.selectedTasks, task],
    })),
  setSelectedTasks: (tasks) => set({ selectedTasks: tasks }),

  // Reset
  resetSelection: () =>
    set({
      selectedModels: [],
      selectedTasks: [],
    }),
}));

// Selectors
export const useActiveBenchmark = () =>
  useBenchmarkStore((state) => state.activeBenchmark);

export const useProgress = () =>
  useBenchmarkStore((state) => state.progress);

export const useSelectedModels = () =>
  useBenchmarkStore((state) => state.selectedModels);

export const useSelectedTasks = () =>
  useBenchmarkStore((state) => state.selectedTasks);
