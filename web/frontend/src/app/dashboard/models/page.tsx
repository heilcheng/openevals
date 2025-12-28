"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  Cpu,
  Settings2,
  Trash2,
  ExternalLink,
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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { modelAPI, type ModelConfigResponse, type ModelTypeInfo } from "@/lib/api";

// Default model types with metadata
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

export default function ModelsPage() {
  const [models, setModels] = useState<ModelConfigResponse[]>([]);
  const [modelTypes, setModelTypes] = useState<ModelTypeInfo[]>(defaultModelTypes);
  const [isLoading, setIsLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newModel, setNewModel] = useState({
    name: "",
    model_type: "",
    size: "",
  });

  useEffect(() => {
    async function fetchData() {
      try {
        const modelsData = await modelAPI.list().catch(() => []);
        setModels(modelsData);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const handleAddModel = async () => {
    if (!newModel.name || !newModel.model_type) return;

    try {
      const created = await modelAPI.create({
        name: newModel.name,
        model_type: newModel.model_type,
        size: newModel.size,
      });
      setModels((prev) => [...prev, created]);
      setNewModel({ name: "", model_type: "", size: "" });
      setShowAddForm(false);
    } catch (error) {
      console.error("Failed to create model:", error);
    }
  };

  const handleDeleteModel = async (id: string) => {
    try {
      await modelAPI.delete(id);
      setModels((prev) => prev.filter((m) => m.id !== id));
    } catch (error) {
      console.error("Failed to delete model:", error);
    }
  };

  const selectedType = modelTypes.find((t) => t.type === newModel.model_type);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Models</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Supported model families and saved configurations
          </p>
        </div>
        <Button
          onClick={() => setShowAddForm(!showAddForm)}
          size="sm"
          className="gap-2"
        >
          <Plus className="h-4 w-4" />
          Add Model
        </Button>
      </div>

      {/* Add Model Form */}
      {showAddForm && (
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Add Configuration</CardTitle>
            <CardDescription className="text-xs">
              Save a model configuration for benchmarking
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className="text-sm font-medium">Name</label>
                <Input
                  placeholder="My Model"
                  value={newModel.name}
                  onChange={(e) =>
                    setNewModel((prev) => ({ ...prev, name: e.target.value }))
                  }
                  className="mt-1.5"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Type</label>
                <Select
                  value={newModel.model_type}
                  onValueChange={(value) =>
                    setNewModel((prev) => ({
                      ...prev,
                      model_type: value,
                      size: "",
                    }))
                  }
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    {modelTypes.map((type) => (
                      <SelectItem key={type.type} value={type.type}>
                        {type.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium">Size</label>
                <Select
                  value={newModel.size}
                  onValueChange={(value) =>
                    setNewModel((prev) => ({ ...prev, size: value }))
                  }
                  disabled={!selectedType}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue placeholder="Select size" />
                  </SelectTrigger>
                  <SelectContent>
                    {selectedType?.sizes.map((size) => (
                      <SelectItem key={size} value={size}>
                        {size}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAddForm(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleAddModel}
                disabled={!newModel.name || !newModel.model_type}
              >
                Save
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Families Table */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Supported Models</CardTitle>
          <CardDescription className="text-xs">
            Open-weight model families with built-in loader support
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="border rounded-md">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left font-medium p-3">Model</th>
                  <th className="text-left font-medium p-3">Type</th>
                  <th className="text-left font-medium p-3">Available Sizes</th>
                  <th className="text-left font-medium p-3 hidden md:table-cell">Source</th>
                </tr>
              </thead>
              <tbody>
                {modelTypes.map((type, index) => (
                  <tr
                    key={type.type}
                    className={index !== modelTypes.length - 1 ? "border-b" : ""}
                  >
                    <td className="p-3 font-medium">{type.name}</td>
                    <td className="p-3">
                      <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                        {type.type}
                      </code>
                    </td>
                    <td className="p-3">
                      <div className="flex flex-wrap gap-1">
                        {type.sizes.slice(0, 5).map((size) => (
                          <Badge
                            key={size}
                            variant="outline"
                            className="text-xs font-normal"
                          >
                            {size}
                          </Badge>
                        ))}
                        {type.sizes.length > 5 && (
                          <Badge variant="outline" className="text-xs font-normal">
                            +{type.sizes.length - 5}
                          </Badge>
                        )}
                      </div>
                    </td>
                    <td className="p-3 hidden md:table-cell">
                      <a
                        href={`https://huggingface.co/models?search=${type.name.toLowerCase()}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
                      >
                        HuggingFace
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Saved Configurations */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Saved Configurations</CardTitle>
          <CardDescription className="text-xs">
            Custom model configurations for quick access
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-12 animate-pulse rounded-md bg-muted/50"
                />
              ))}
            </div>
          ) : models.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <Settings2 className="h-8 w-8 text-muted-foreground/50" />
              <p className="mt-3 text-sm text-muted-foreground">
                No saved configurations
              </p>
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onClick={() => setShowAddForm(true)}
              >
                Add configuration
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.id}
                  className="group flex items-center justify-between rounded-md border p-3"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted">
                      <Cpu className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{model.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {model.model_type}
                        {model.size && ` · ${model.size}`}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 opacity-0 group-hover:opacity-100"
                    onClick={() => handleDeleteModel(model.id)}
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
