"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface ShimmerButtonProps extends HTMLMotionProps<"button"> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
  className?: string;
  children: React.ReactNode;
}

export function ShimmerButton({
  shimmerColor = "rgba(255, 255, 255, 0.2)",
  shimmerSize = "0.1em",
  shimmerDuration = "2s",
  borderRadius = "0.5rem",
  background = "linear-gradient(135deg, hsl(var(--primary)) 0%, hsl(var(--accent)) 100%)",
  className,
  children,
  ...props
}: ShimmerButtonProps) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "group relative inline-flex items-center justify-center overflow-hidden px-6 py-3 font-medium text-primary-foreground transition-all",
        className
      )}
      style={{
        borderRadius,
        background,
      }}
      {...props}
    >
      {/* Shimmer overlay */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ borderRadius }}
      >
        <div
          className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/20 to-transparent"
          style={{
            animationDuration: shimmerDuration,
          }}
        />
      </div>

      {/* Glow effect on hover */}
      <div
        className="absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{
          borderRadius,
          boxShadow: `0 0 20px hsla(var(--primary), 0.5), 0 0 40px hsla(var(--primary), 0.3)`,
        }}
      />

      {/* Content */}
      <span className="relative z-10 flex items-center gap-2">{children}</span>
    </motion.button>
  );
}

interface GlowingButtonProps extends HTMLMotionProps<"button"> {
  children: React.ReactNode;
  className?: string;
}

export function GlowingButton({
  children,
  className,
  ...props
}: GlowingButtonProps) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "relative inline-flex items-center justify-center overflow-hidden rounded-lg bg-gradient-to-r from-primary to-accent px-6 py-3 font-medium text-primary-foreground",
        className
      )}
      {...props}
    >
      {/* Animated border */}
      <motion.span
        className="absolute inset-0 rounded-lg"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)",
          backgroundSize: "200% 100%",
        }}
        animate={{
          backgroundPosition: ["100% 0%", "-100% 0%"],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Glow */}
      <span
        className="absolute inset-0 rounded-lg opacity-50 blur-lg"
        style={{
          background:
            "linear-gradient(135deg, hsl(var(--primary)), hsl(var(--accent)))",
        }}
      />

      <span className="relative">{children}</span>
    </motion.button>
  );
}
