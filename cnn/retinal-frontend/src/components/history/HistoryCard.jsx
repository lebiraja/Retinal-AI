import { motion } from 'framer-motion';
import { Clock, TrendingUp, Trash2, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatConfidence, formatTimestamp, getRiskLevel } from '@/utils/formatters';

/**
 * Single history entry card
 */
export function HistoryCard({ entry, index, onRemove }) {
  const risk = getRiskLevel(entry.confidence);

  const iconMap = {
    High: <AlertTriangle className="h-3.5 w-3.5" />,
    Moderate: <TrendingUp className="h-3.5 w-3.5" />,
    Low: <CheckCircle2 className="h-3.5 w-3.5" />,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.3 }}
    >
      <Card className="group hover:shadow-lg transition-shadow">
        <CardContent className="p-5">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0 space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-semibold text-foreground truncate">
                  {entry.disease}
                </h3>
                <Badge variant={risk.color} className="shrink-0">
                  {iconMap[risk.level]}
                  <span className="ml-1">{risk.level}</span>
                </Badge>
              </div>

              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  {formatConfidence(entry.confidence)}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatTimestamp(entry.timestamp)}
                </span>
              </div>

              {/* Mini confidence bar */}
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${Math.round(entry.confidence * 100)}%` }}
                />
              </div>
            </div>

            <Button
              variant="ghost"
              size="icon"
              className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive shrink-0"
              onClick={() => onRemove(entry.id)}
              aria-label="Remove entry"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
