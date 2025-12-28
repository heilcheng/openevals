const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api/v1";

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }

  return response.json();
}

// Benchmarks
export const benchmarkAPI = {
  list: (params?: { skip?: number; limit?: number; status?: string }) => {
    const query = new URLSearchParams();
    if (params?.skip) query.set("offset", String(params.skip));
    if (params?.limit) query.set("limit", String(params.limit));
    if (params?.status) query.set("status", params.status);
    return fetchAPI<BenchmarkRunResponse[]>(`/benchmarks?${query}`).then(
      (items) => ({
        items,
        total: items.length,
        skip: params?.skip || 0,
        limit: params?.limit || 50,
      })
    );
  },

  get: (id: string) => fetchAPI<BenchmarkRunResponse>(`/benchmarks/${id}`),

  create: (data: BenchmarkCreateRequest) =>
    fetchAPI<BenchmarkRunResponse>("/benchmarks", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  cancel: (id: string) =>
    fetchAPI<{ status: string; run_id: string }>(`/benchmarks/${id}/cancel`, {
      method: "POST",
    }),

  delete: (id: string) =>
    fetchAPI<{ status: string }>(`/benchmarks/${id}`, {
      method: "DELETE",
    }),

  stats: () =>
    fetchAPI<{
      total_runs: number;
      completed_runs: number;
      running_runs: number;
      saved_models: number;
      unique_models_tested: number;
      available_tasks: number;
    }>("/benchmarks/stats"),
};

// Models
export const modelAPI = {
  list: () => fetchAPI<ModelConfigResponse[]>("/models"),

  get: (id: string) => fetchAPI<ModelConfigResponse>(`/models/${id}`),

  create: (data: ModelConfigCreate) =>
    fetchAPI<ModelConfigResponse>("/models", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<ModelConfigCreate>) =>
    fetchAPI<ModelConfigResponse>(`/models/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    fetchAPI<{ status: string; id: string }>(`/models/${id}`, {
      method: "DELETE",
    }),

  types: () =>
    fetchAPI<{ types: string[]; defaults: Record<string, Record<string, unknown>> }>(
      "/models/types"
    ).then((data) =>
      data.types.map((type) => ({
        type,
        name: type.charAt(0).toUpperCase() + type.slice(1),
        sizes: ["2b", "7b", "8b", "13b"],
        default_size: "7b",
      }))
    ),
};

// Tasks
export const taskAPI = {
  list: () => fetchAPI<TaskInfo[]>("/tasks"),
  get: (type: string) => fetchAPI<TaskInfo>(`/tasks/${type}`),
};

// Results
export const resultAPI = {
  list: (params?: {
    run_id?: string;
    model_name?: string;
    task_name?: string;
    limit?: number;
  }) => {
    const query = new URLSearchParams();
    if (params?.run_id) query.set("run_id", params.run_id);
    if (params?.model_name) query.set("model_name", params.model_name);
    if (params?.task_name) query.set("task_name", params.task_name);
    if (params?.limit) query.set("limit", String(params.limit));
    // For now, return empty array - results come from benchmark detail
    return Promise.resolve<TaskResultResponse[]>([]);
  },

  get: (id: string) => fetchAPI<TaskResultResponse>(`/results/${id}`),

  leaderboard: () =>
    fetchAPI<LeaderboardEntry[]>("/benchmarks/leaderboard").then((entries) => ({
      entries,
      tasks: entries.length > 0 ? Object.keys(entries[0].task_scores) : [],
      total_models: entries.length,
    })),

  compare: (models: string[], tasks?: string[]) => {
    const query = new URLSearchParams();
    query.set("models", models.join(","));
    if (tasks?.length) query.set("tasks", tasks.join(","));
    return fetchAPI<{
      models: string[];
      tasks: string[];
      scores: Record<string, Record<string, number>>;
    }>(`/results/compare?${query}`);
  },
};

// Types
export interface BenchmarkRunResponse {
  id: string;
  name: string;
  description?: string;
  status: string;
  config: Record<string, unknown>;
  error_message?: string;
  progress_percent: number;
  current_model?: string;
  current_task?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at?: string;
  task_results?: TaskResultResponse[];
}

export interface BenchmarkCreateRequest {
  name: string;
  description?: string;
  models: string[];
  tasks: string[];
  config?: Record<string, unknown>;
  model_configs?: Record<string, Record<string, unknown>>;
  task_configs?: Record<string, Record<string, unknown>>;
}

export interface ModelConfigResponse {
  id: string;
  name: string;
  model_type: string;
  size?: string;
  variant?: string;
  config: Record<string, unknown>;
  is_public: boolean;
  created_at: string;
  updated_at?: string;
}

export interface ModelConfigCreate {
  name: string;
  model_type: string;
  size?: string;
  variant?: string;
  config?: Record<string, unknown>;
  is_public?: boolean;
}

export interface ModelTypeInfo {
  type: string;
  name: string;
  sizes: string[];
  default_size: string;
}

export interface TaskInfo {
  type: string;
  name: string;
  description: string;
  metrics: string[];
  default_config?: Record<string, unknown>;
}

export interface TaskResultResponse {
  id: string;
  run_id: string;
  model_name: string;
  task_name: string;
  status: string;
  overall?: Record<string, number>;
  details?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  errors?: string[];
  duration_seconds?: string;
  started_at?: string;
  completed_at?: string;
  created_at?: string;
}

export interface LeaderboardEntry {
  rank: number;
  model_name: string;
  average_score: number;
  task_scores: Record<string, number>;
  total_runs: number;
  runs_count?: number;
}
