"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Plus,
  Cpu,
  Settings2,
  Trash2,
  Sparkles,
  Zap,
  Server,
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

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const item = {
  hidden: { opacity: 0, scale: 0.95 },
  show: { opacity: 1, scale: 1 },
};

const modelIcons: Record<string, React.ReactNode> = {
  gemma: <Sparkles className="h-5 w-5" />,
  mistral: <Zap className="h-5 w-5" />,
  llama: <Server className="h-5 w-5" />,
  huggingface: <Cpu className="h-5 w-5" />,
};

const modelColors: Record<string, string> = {
  gemma: "from-blue-500 to-cyan-500",
  mistral: "from-orange-500 to-amber-500",
  llama: "from-purple-500 to-pink-500",
  huggingface: "from-yellow-500 to-orange-500",
};

export default function ModelsPage() {
  const [models, setModels] = useState<ModelConfigResponse[]>([]);
  const [modelTypes, setModelTypes] = useState<ModelTypeInfo[]>([]);
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
        const [modelsData, typesData] = await Promise.all([
          modelAPI.list(),
          modelAPI.types(),
        ]);
        setModels(modelsData);
        setModelTypes(typesData);
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
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            Manage your model configurations
          </p>
        </div>
        <Button
          onClick={() => setShowAddForm(!showAddForm)}
          className="gap-2 gradient-primary text-primary-foreground"
        >
          <Plus className="h-4 w-4" />
          Add Model
        </Button>
      </div>

      {/* Add Model Form */}
      {showAddForm && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
        >
          <Card className="border-primary/50">
            <CardHeader>
              <CardTitle>Add New Model</CardTitle>
              <CardDescription>
                Configure a new model for benchmarking
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <label className="text-sm font-medium">Model Name</label>
                  <Input
                    placeholder="My Gemma Model"
                    value={newModel.name}
                    onChange={(e) =>
                      setNewModel((prev) => ({ ...prev, name: e.target.value }))
                    }
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Model Type</label>
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
                    <SelectTrigger className="mt-1">
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
                    <SelectTrigger className="mt-1">
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
                  onClick={() => setShowAddForm(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleAddModel}
                  disabled={!newModel.name || !newModel.model_type}
                >
                  Add Model
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Available Model Types */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Model Types</CardTitle>
          <CardDescription>
            Built-in support for popular model families
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {modelTypes.map((type, index) => (
              <motion.div
                key={type.type}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="group relative overflow-hidden rounded-lg border p-4 transition-all hover:border-primary/50"
              >
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${
                    modelColors[type.type] || "from-gray-500 to-gray-600"
                  } opacity-0 transition-opacity group-hover:opacity-5`}
                />
                <div className="relative">
                  <div
                    className={`inline-flex rounded-lg bg-gradient-to-br ${
                      modelColors[type.type] || "from-gray-500 to-gray-600"
                    } p-2 text-white`}
                  >
                    {modelIcons[type.type] || <Cpu className="h-5 w-5" />}
                  </div>
                  <h3 className="mt-3 font-semibold">{type.name}</h3>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {type.sizes.map((size) => (
                      <Badge
                        key={size}
                        variant="secondary"
                        className="text-xs"
                      >
                        {size}
                      </Badge>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Saved Models */}
      <Card>
        <CardHeader>
          <CardTitle>Saved Configurations</CardTitle>
          <CardDescription>
            Your custom model configurations for benchmarking
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-32 animate-pulse rounded-lg bg-muted/50"
                />
              ))}
            </div>
          ) : models.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="rounded-full bg-muted p-4">
                <Settings2 className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="mt-4 font-medium">No saved models</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Add a model configuration to get started
              </p>
            </div>
          ) : (
            <motion.div
              variants={container}
              initial="hidden"
              animate="show"
              className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
            >
              {models.map((model) => (
                <motion.div
                  key={model.id}
                  variants={item}
                  className="group relative rounded-lg border p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={`rounded-lg bg-gradient-to-br ${
                          modelColors[model.model_type] ||
                          "from-gray-500 to-gray-600"
                        } p-2 text-white`}
                      >
                        {modelIcons[model.model_type] || (
                          <Cpu className="h-4 w-4" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium">{model.name}</p>
                        <p className="text-sm text-muted-foreground capitalize">
                          {model.model_type}
                          {model.size && ` • ${model.size}`}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="opacity-0 transition-opacity group-hover:opacity-100"
                      onClick={() => handleDeleteModel(model.id)}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    Created {new Date(model.created_at).toLocaleDateString()}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
