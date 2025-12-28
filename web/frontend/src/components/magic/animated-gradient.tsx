"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface AnimatedGradientProps {
  className?: string;
  children?: React.ReactNode;
  containerClassName?: string;
}

export function AnimatedGradient({
  className,
  children,
  containerClassName,
}: AnimatedGradientProps) {
  return (
    <div className={cn("relative overflow-hidden", containerClassName)}>
      <motion.div
        className={cn(
          "absolute inset-0 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%]",
          className
        )}
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 5,
          ease: "linear",
          repeat: Infinity,
        }}
      />
      {children && <div className="relative z-10">{children}</div>}
    </div>
  );
}

export function AnimatedGradientText({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.span
      className={cn(
        "bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_auto] bg-clip-text text-transparent",
        className
      )}
      animate={{
        backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
      }}
      transition={{
        duration: 3,
        ease: "linear",
        repeat: Infinity,
      }}
    >
      {children}
    </motion.span>
  );
}

export function AnimatedGradientBorder({
  children,
  className,
  borderWidth = 2,
}: {
  children: React.ReactNode;
  className?: string;
  borderWidth?: number;
}) {
  return (
    <div className={cn("relative rounded-lg", className)}>
      <motion.div
        className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%]"
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 3,
          ease: "linear",
          repeat: Infinity,
        }}
      />
      <div
        className="relative rounded-lg bg-background"
        style={{ margin: borderWidth }}
      >
        {children}
      </div>
    </div>
  );
}
