import { motion } from 'framer-motion';
import { formatConfidence, getRiskLevel } from '@/utils/formatters';
import { cn } from '@/utils/cn';

const barColors = {
  destructive: 'bg-red-500',
  warning: 'bg-amber-500',
  success: 'bg-emerald-500',
};

const textColors = {
  destructive: 'text-red-600 dark:text-red-400',
  warning: 'text-amber-600 dark:text-amber-400',
  success: 'text-emerald-600 dark:text-emerald-400',
};

/**
 * Animated horizontal confidence bar for a single prediction
 */
export function ConfidenceBar({ label, value, index = 0, isTop = false }) {
  const risk = getRiskLevel(value);
  const pct = Math.round(value * 100);

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.06, duration: 0.35 }}
      className={cn(
        'rounded-xl p-3 transition-colors',
        isTop && 'bg-primary/5 border border-primary/10',
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span
          className={cn(
            'text-sm font-medium',
            isTop ? 'text-foreground font-semibold' : 'text-foreground',
          )}
        >
          {label}
        </span>
        <span className={cn('text-sm font-bold tabular-nums', textColors[risk.color])}>
          {formatConfidence(value)}
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: index * 0.06 + 0.15, duration: 0.7, ease: 'easeOut' }}
          className={cn('h-full rounded-full', barColors[risk.color])}
        />
      </div>
    </motion.div>
  );
}
