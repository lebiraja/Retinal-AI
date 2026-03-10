import { motion, AnimatePresence } from 'framer-motion';
import { Eye, RotateCcw, WifiOff } from 'lucide-react';
import { usePrediction } from '@/hooks/usePrediction';
import { ImageUpload } from '@/components/prediction/ImageUpload';
import { ResultsPanel } from '@/components/prediction/ResultsPanel';
import { PageTransition } from '@/components/layout/PageTransition';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';

export default function Predict() {
  const { predict, data, isLoading, isError, error, reset } = usePrediction();

  const handleSubmit = (file) => {
    predict(file);
  };

  return (
    <PageTransition>
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 border border-primary/20">
              <Eye className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-black text-foreground">Image Analysis</h1>
              <p className="text-sm text-muted-foreground">
                Upload an eye image for AI-powered disease screening
              </p>
            </div>
          </div>
          {(data || isError) && (
            <Button variant="outline" onClick={reset}>
              <RotateCcw className="h-4 w-4" />
              New Analysis
            </Button>
          )}
        </div>

        <div className="grid lg:grid-cols-5 gap-6">
          {/* Upload panel */}
          <div className="lg:col-span-2">
            <ImageUpload onSubmit={handleSubmit} isLoading={isLoading} />
          </div>

          {/* Results panel */}
          <div className="lg:col-span-3">
            <AnimatePresence mode="wait">
              {/* Loading */}
              {isLoading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  <Skeleton className="h-40 w-full rounded-2xl" />
                  <Skeleton className="h-28 w-full rounded-2xl" />
                  <Skeleton className="h-72 w-full rounded-2xl" />
                </motion.div>
              )}

              {/* Error */}
              {isError && !isLoading && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center gap-4 py-20 text-center"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-destructive/10 border border-destructive/20">
                    <WifiOff className="h-8 w-8 text-destructive" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">
                      Analysis Failed
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                      {error?.message || 'Unable to connect to the AI backend. Please try again.'}
                    </p>
                  </div>
                  <Button variant="outline" onClick={reset}>
                    <RotateCcw className="h-4 w-4" />
                    Try Again
                  </Button>
                </motion.div>
              )}

              {/* Results */}
              {data && !isLoading && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <ResultsPanel data={data} />
                </motion.div>
              )}

              {/* Empty state */}
              {!data && !isLoading && !isError && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center gap-4 py-20 text-center"
                >
                  <motion.div
                    animate={{ y: [0, -6, 0] }}
                    transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                    className="flex h-20 w-20 items-center justify-center rounded-2xl bg-muted border border-border"
                  >
                    <Eye className="h-10 w-10 text-muted-foreground/40" />
                  </motion.div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">
                      No analysis yet
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1 max-w-xs">
                      Upload a fundus image and click "Start Analysis" to see
                      AI-powered disease predictions here.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
