import { motion } from 'framer-motion';

/**
 * Retinal scan animation overlay.
 * Rendered over the image preview while analysis is in progress.
 * Mimics a barcode / retinal scanner with a sweeping laser line.
 */
export function ScanOverlay() {
  return (
    <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
      {/* Dim backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Corner brackets — scanner viewfinder */}
      <CornerBrackets />

      {/* Horizontal scan grid lines (static, low opacity) */}
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="absolute left-0 right-0 h-px bg-green-400/10"
          style={{ top: `${15 + i * 14}%` }}
        />
      ))}

      {/* Sweeping laser line */}
      <motion.div
        className="absolute left-[6%] right-[6%] h-[2px] rounded-full"
        style={{
          background:
            'linear-gradient(90deg, transparent 0%, #4ade80 20%, #86efac 50%, #4ade80 80%, transparent 100%)',
          boxShadow: '0 0 10px 3px rgba(74,222,128,0.65), 0 0 2px 1px rgba(74,222,128,0.9)',
        }}
        animate={{ top: ['8%', '88%'] }}
        transition={{
          duration: 1.8,
          repeat: Infinity,
          repeatType: 'reverse',
          ease: 'easeInOut',
        }}
      />

      {/* Glow trail below the line */}
      <motion.div
        className="absolute left-[6%] right-[6%] h-8 rounded-full"
        style={{
          background:
            'linear-gradient(180deg, rgba(74,222,128,0.18) 0%, transparent 100%)',
        }}
        animate={{ top: ['8%', '88%'] }}
        transition={{
          duration: 1.8,
          repeat: Infinity,
          repeatType: 'reverse',
          ease: 'easeInOut',
        }}
      />

      {/* Bottom status bar */}
      <div className="absolute bottom-3 inset-x-0 flex items-center justify-center gap-2.5">
        <motion.span
          animate={{ opacity: [1, 0.2, 1] }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'easeInOut' }}
          className="h-1.5 w-1.5 rounded-full bg-green-400"
        />
        <span className="text-[11px] font-mono font-bold tracking-[0.2em] text-green-400 uppercase select-none">
          Scanning Retina
        </span>
        <motion.span
          animate={{ opacity: [1, 0.2, 1] }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'easeInOut', delay: 0.45 }}
          className="h-1.5 w-1.5 rounded-full bg-green-400"
        />
      </div>
    </div>
  );
}

/** Four corner brackets forming a scanner viewfinder */
function CornerBrackets() {
  const base = 'absolute h-6 w-6 border-green-400';
  const w = 'border-[2.5px]';
  return (
    <>
      <div className={`${base} ${w} top-2.5 left-2.5  border-t border-l rounded-tl`} />
      <div className={`${base} ${w} top-2.5 right-2.5 border-t border-r rounded-tr`} />
      <div className={`${base} ${w} bottom-2.5 left-2.5  border-b border-l rounded-bl`} />
      <div className={`${base} ${w} bottom-2.5 right-2.5 border-b border-r rounded-br`} />
    </>
  );
}
