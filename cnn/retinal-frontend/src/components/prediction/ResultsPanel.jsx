import { motion } from 'framer-motion';
import {
  Activity,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ShieldCheck,
  Info,
  Stethoscope,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ConfidenceBar } from './ConfidenceBar';
import { PredictionChart } from '@/components/charts/PredictionChart';
import { formatConfidence, formatProcessingTime } from '@/utils/formatters';

/** Map backend risk levels to badge styling */
function getRiskBadge(level) {
  switch (level?.toUpperCase()) {
    case 'HIGH':
      return { variant: 'destructive', icon: ShieldAlert, label: 'High Risk' };
    case 'MODERATE':
      return { variant: 'warning', icon: AlertTriangle, label: 'Moderate Risk' };
    case 'LOW':
    default:
      return { variant: 'success', icon: ShieldCheck, label: 'Low Risk' };
  }
}

const riskColorMap = {
  destructive: 'text-red-600 dark:text-red-400',
  warning: 'text-amber-600 dark:text-amber-400',
  success: 'text-emerald-600 dark:text-emerald-400',
};

/**
 * Full results panel — shows after a successful prediction
 */
export function ResultsPanel({ data }) {
  if (!data?.predictions?.length) {
    /* No diseases detected — show a clean "healthy" card */
    if (data && !data.diseaseRisk) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-5"
        >
          <Card className="overflow-hidden">
            <div className="bg-gradient-to-r from-emerald-500/10 via-emerald-500/5 to-transparent p-6 text-center">
              <CheckCircle2 className="h-12 w-12 text-emerald-500 mx-auto mb-3" />
              <h3 className="text-xl font-bold text-foreground">No Disease Detected</h3>
              <p className="text-sm text-muted-foreground mt-1">
                The AI did not detect any retinal conditions above the
                {data.threshold ? ` ${(data.threshold * 100).toFixed(0)}%` : ''} confidence threshold.
              </p>
              <p className="text-xs text-muted-foreground mt-3">
                Processed in {formatProcessingTime(data.processing_time)}
              </p>
            </div>
          </Card>
          {data.disclaimer && <DisclaimerBox text={data.disclaimer} />}
        </motion.div>
      );
    }
    return null;
  }

  const topPrediction = data.predictions[0];
  const riskBadge = getRiskBadge(data.riskLevel);
  const RiskIcon = riskBadge.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-5"
    >
      {/* Top result banner */}
      <Card className="overflow-hidden">
        <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                <h3 className="text-sm font-medium text-muted-foreground">
                  Top Prediction
                </h3>
              </div>
              <p className="text-2xl font-bold text-foreground">
                {topPrediction.fullName || topPrediction.disease}
              </p>
              <div className="flex items-center gap-3 flex-wrap">
                <Badge variant={riskBadge.variant}>
                  <RiskIcon className="h-3 w-3 mr-1" />
                  {riskBadge.label}
                </Badge>
                <span className={`text-lg font-bold ${riskColorMap[riskBadge.variant]}`}>
                  {formatConfidence(topPrediction.confidence)}
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-2 text-right">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span className="text-sm">{formatProcessingTime(data.processing_time)}</span>
              </div>
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">{data.numDetected ?? data.predictions.length} condition{(data.numDetected ?? data.predictions.length) !== 1 ? 's' : ''} detected</span>
              </div>
            </div>
          </div>
        </div>

        {/* Animated progress bar for top prediction */}
        <CardContent className="pt-0 -mt-1">
          <ConfidenceBar
            label={topPrediction.fullName || topPrediction.disease}
            value={topPrediction.confidence}
            isTop
          />
        </CardContent>
      </Card>

      {/* Advisory */}
      {data.advisory && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Stethoscope className="h-4 w-4 text-primary" />
              Clinical Advisory
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
              {data.advisory}
            </p>
          </CardContent>
        </Card>
      )}

      {/* All predictions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            All Predictions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {data.predictions.map((pred, i) => (
            <ConfidenceBar
              key={pred.disease}
              label={pred.fullName || pred.disease}
              value={pred.confidence}
              index={i}
            />
          ))}
        </CardContent>
      </Card>

      {/* Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            Confidence Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <PredictionChart predictions={data.predictions} />
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <DisclaimerBox text={data.disclaimer} />
    </motion.div>
  );
}

/** Reusable disclaimer section */
function DisclaimerBox({ text }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl bg-muted/50 border border-border px-4 py-3">
      <Info className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
      <p className="text-xs text-muted-foreground leading-relaxed">
        <span className="font-semibold text-foreground">Medical Disclaimer:</span>{' '}
        {text ||
          'This system is for screening and educational purposes only. Results should not be used as a clinical diagnosis. Always consult a qualified ophthalmologist.'}
      </p>
    </div>
  );
}
