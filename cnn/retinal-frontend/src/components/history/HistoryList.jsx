import { AnimatePresence, motion } from 'framer-motion';
import { History, Trash2, ClipboardList } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { HistoryCard } from './HistoryCard';
import useHistoryStore from '@/store/useHistoryStore';

/**
 * Full history list with empty state
 */
export function HistoryList() {
  const { history, clearHistory, removeEntry } = useHistoryStore();

  /* Empty state */
  if (history.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col items-center justify-center gap-4 py-20 text-center"
      >
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-muted">
          <ClipboardList className="h-10 w-10 text-muted-foreground/50" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">No history yet</h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">
            Your last 5 predictions will appear here. Run an analysis to get started.
          </p>
        </div>
      </motion.div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-bold text-foreground">
            Recent Analyses
          </h2>
          <span className="text-xs text-muted-foreground">
            ({history.length}/5)
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={clearHistory}
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="h-4 w-4 mr-1" />
          Clear All
        </Button>
      </div>

      <div className="space-y-3">
        <AnimatePresence>
          {history.map((entry, i) => (
            <HistoryCard
              key={entry.id}
              entry={entry}
              index={i}
              onRemove={removeEntry}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
