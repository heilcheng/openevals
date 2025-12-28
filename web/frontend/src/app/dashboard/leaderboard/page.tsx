"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Trophy, Medal, Award, TrendingUp } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { resultAPI, type LeaderboardEntry } from "@/lib/api";

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) {
    return (
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-yellow-400 to-amber-500 shadow-lg shadow-yellow-500/20">
        <Trophy className="h-4 w-4 text-white" />
      </div>
    );
  }
  if (rank === 2) {
    return (
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-gray-300 to-gray-400 shadow-lg shadow-gray-400/20">
        <Medal className="h-4 w-4 text-white" />
      </div>
    );
  }
  if (rank === 3) {
    return (
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-amber-600 to-amber-700 shadow-lg shadow-amber-600/20">
        <Award className="h-4 w-4 text-white" />
      </div>
    );
  }
  return (
    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-sm font-medium">
      {rank}
    </div>
  );
}

function ScoreBar({ score, maxScore = 1 }: { score: number; maxScore?: number }) {
  const percentage = (score / maxScore) * 100;
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="h-full bg-gradient-to-r from-primary to-purple-500"
      />
    </div>
  );
}

export default function LeaderboardPage() {
  const [leaderboard, setLeaderboard] = useState<{
    entries: LeaderboardEntry[];
    tasks: string[];
    total_models: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTask, setSelectedTask] = useState<string>("all");

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        const data = await resultAPI.leaderboard();
        setLeaderboard(data);
      } catch (error) {
        console.error("Failed to fetch leaderboard:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchLeaderboard();
  }, []);

  const getSortedEntries = () => {
    if (!leaderboard) return [];

    if (selectedTask === "all") {
      return [...leaderboard.entries].sort(
        (a, b) => b.average_score - a.average_score
      );
    }

    return [...leaderboard.entries]
      .filter((e) => selectedTask in e.task_scores)
      .sort(
        (a, b) =>
          (b.task_scores[selectedTask] || 0) -
          (a.task_scores[selectedTask] || 0)
      );
  };

  const getScore = (entry: LeaderboardEntry) => {
    if (selectedTask === "all") {
      return entry.average_score;
    }
    return entry.task_scores[selectedTask] || 0;
  };

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  const sortedEntries = getSortedEntries();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Leaderboard</h1>
          <p className="text-muted-foreground">
            Model rankings based on benchmark performance
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <TrendingUp className="h-4 w-4" />
          {leaderboard?.total_models || 0} models ranked
        </div>
      </div>

      {/* Top 3 Podium */}
      {sortedEntries.length >= 3 && (
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 0, 2].map((podiumIndex) => {
            const entry = sortedEntries[podiumIndex];
            if (!entry) return null;
            const rank = podiumIndex === 1 ? 1 : podiumIndex === 0 ? 2 : 3;
            const score = getScore(entry);

            return (
              <motion.div
                key={entry.model_name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: rank * 0.1 }}
                className={rank === 1 ? "md:order-2" : rank === 2 ? "md:order-1" : "md:order-3"}
              >
                <Card
                  className={`relative overflow-hidden ${
                    rank === 1 ? "border-yellow-500/50 bg-yellow-500/5" : ""
                  }`}
                >
                  {rank === 1 && (
                    <div className="absolute inset-0 bg-gradient-to-br from-yellow-500/10 via-transparent to-transparent" />
                  )}
                  <CardContent className="relative flex flex-col items-center p-6 text-center">
                    <RankBadge rank={rank} />
                    <h3 className="mt-4 text-lg font-semibold">
                      {entry.model_name}
                    </h3>
                    <p className="mt-2 text-3xl font-bold">
                      {(score * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {entry.runs_count} runs
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Task Filter */}
      {leaderboard && leaderboard.tasks.length > 0 && (
        <Tabs value={selectedTask} onValueChange={setSelectedTask}>
          <TabsList>
            <TabsTrigger value="all">Overall</TabsTrigger>
            {leaderboard.tasks.map((task) => (
              <TabsTrigger key={task} value={task} className="capitalize">
                {task.replace(/_/g, " ")}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      )}

      {/* Full Leaderboard */}
      <Card>
        <CardHeader>
          <CardTitle>Full Rankings</CardTitle>
          <CardDescription>
            {selectedTask === "all"
              ? "Average performance across all tasks"
              : `Performance on ${selectedTask.replace(/_/g, " ")}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sortedEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12">
              <Trophy className="h-12 w-12 text-muted-foreground" />
              <h3 className="mt-4 text-lg font-medium">No rankings yet</h3>
              <p className="mt-2 text-sm text-muted-foreground">
                Complete some benchmarks to see the leaderboard
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {sortedEntries.map((entry, index) => {
                const score = getScore(entry);
                return (
                  <motion.div
                    key={entry.model_name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className="flex items-center gap-4 rounded-lg border p-4"
                  >
                    <RankBadge rank={index + 1} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium truncate">
                          {entry.model_name}
                        </h4>
                        <span className="ml-4 text-lg font-bold">
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-2">
                        <ScoreBar score={score} />
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {Object.entries(entry.task_scores).map(([task, taskScore]) => (
                          <Badge
                            key={task}
                            variant={task === selectedTask ? "default" : "outline"}
                            className="text-xs"
                          >
                            {task}: {(taskScore * 100).toFixed(0)}%
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
