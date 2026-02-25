import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const COLORS = ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#ef4444'];

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { disease, confidence } = payload[0].payload;

  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-lg">
      <p className="text-sm font-semibold text-foreground">{disease}</p>
      <p className="text-xs text-muted-foreground">
        Confidence: <span className="font-bold text-primary">{(confidence * 100).toFixed(1)}%</span>
      </p>
    </div>
  );
}

/**
 * Bar chart showing all prediction confidences
 */
export function PredictionChart({ predictions }) {
  if (!predictions?.length) return null;

  const data = predictions.map((p) => ({
    disease: p.fullName || p.disease,
    confidence: p.confidence,
    pct: Math.round(p.confidence * 100),
  }));

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border/50" />
          <XAxis
            dataKey="disease"
            tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
            angle={-35}
            textAnchor="end"
            interval={0}
            height={60}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'hsl(var(--accent))' }} />
          <Bar dataKey="pct" radius={[6, 6, 0, 0]} maxBarSize={48}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
