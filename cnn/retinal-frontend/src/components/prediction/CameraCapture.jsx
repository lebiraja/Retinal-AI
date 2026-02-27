import { useRef, useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Camera, RefreshCw, AlertCircle, FlipHorizontal, VideoOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/utils/cn';

/**
 * Live camera feed component.
 *
 * Flow:
 *   1. Mount → show "Start Camera" button (no auto-start, avoids race + permission UX)
 *   2. User clicks → request getUserMedia → stream into <video>
 *   3. video onCanPlay → set ready=true → show viewfinder overlay
 *   4. Capture Photo → draw to canvas → emit File via onCapture()
 *   5. Unmount → stop all tracks
 *
 * Props:
 *   onCapture(file: File) — called with the captured JPEG file
 *   disabled             — disables capture button while analysis runs
 */
export function CameraCapture({ onCapture, disabled }) {
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const streamRef  = useRef(null);

  const [phase, setPhase]             = useState('idle');   // 'idle' | 'starting' | 'live' | 'error'
  const [errorMsg, setErrorMsg]       = useState('');
  const [facingMode, setFacingMode]   = useState('user');   // desktop has no 'environment' usually

  /* ── Stop stream helper ──────────────────────────────────────── */
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  /* ── Cleanup on unmount ──────────────────────────────────────── */
  useEffect(() => () => stopStream(), [stopStream]);

  /* ── Start camera ────────────────────────────────────────────── */
  const startCamera = useCallback(async (mode) => {
    stopStream();
    setPhase('starting');
    setErrorMsg('');

    // Require secure context (HTTPS or localhost)
    if (!navigator?.mediaDevices?.getUserMedia) {
      setErrorMsg(
        'Camera API unavailable — this page must be served over HTTPS or localhost (http://localhost:5173).',
      );
      setPhase('error');
      return;
    }

    // Try constraints from most specific → most permissive
    const constraintSets = [
      { video: { facingMode: mode, width: { ideal: 1280 }, height: { ideal: 720 } } },
      { video: { facingMode: mode } },
      { video: true },
    ];

    let stream = null;
    let lastErr = null;

    for (const c of constraintSets) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(c);
        break;
      } catch (e) {
        lastErr = e;
        console.warn('[CameraCapture] getUserMedia failed with', c, '→', e.name, e.message);
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') break;
      }
    }

    if (!stream) {
      const name = lastErr?.name ?? 'UnknownError';
      const msgMap = {
        NotAllowedError:        'Permission denied. Click the 🔒 icon in your browser address bar, allow Camera access, then retry.',
        PermissionDeniedError:  'Permission denied. Click the 🔒 icon in your browser address bar, allow Camera access, then retry.',
        NotFoundError:          'No camera device found on this machine.',
        DevicesNotFoundError:   'No camera device found on this machine.',
        NotReadableError:       'Camera is currently used by another application. Close it and retry.',
        TrackStartError:        'Camera is currently used by another application. Close it and retry.',
        OverconstrainedError:   'Camera does not support the requested settings. Retry — it will use simpler settings.',
      };
      setErrorMsg(msgMap[name] ?? `Camera error: ${name} — ${lastErr?.message ?? 'unknown reason'}.`);
      setPhase('error');
      return;
    }

    streamRef.current = stream;

    // Attach stream to video element AFTER it is mounted
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    } else {
      // Edge case: ref not ready yet — wait one tick
      await new Promise((r) => setTimeout(r, 80));
      if (videoRef.current) videoRef.current.srcObject = stream;
    }
    // phase → 'live' is set by onCanPlay on the <video> element
  }, [stopStream]);

  /* ── Flip camera ─────────────────────────────────────────────── */
  const flipCamera = () => {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    startCamera(next);
  };

  /* ── Capture snapshot → File ─────────────────────────────────── */
  const capture = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || phase !== 'live') return;

    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        onCapture(new File([blob], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' }));
      },
      'image/jpeg',
      0.92,
    );
  };

  /* ── Idle: start button ──────────────────────────────────────── */
  if (phase === 'idle') {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-border bg-muted/30 p-10 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
          <Camera className="h-8 w-8 text-primary" />
        </div>
        <div>
          <p className="text-sm font-semibold text-foreground">Use Your Camera</p>
          <p className="text-xs text-muted-foreground mt-1 max-w-xs">
            Capture a retinal fundus image directly with your system camera.
            Your browser will ask for permission.
          </p>
        </div>
        <Button onClick={() => startCamera(facingMode)} disabled={disabled}>
          <Camera className="h-4 w-4" />
          Start Camera
        </Button>
      </div>
    );
  }

  /* ── Error ───────────────────────────────────────────────────── */
  if (phase === 'error') {
    return (
      <div className="flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-destructive/40 bg-destructive/5 p-8 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-destructive/10">
          <AlertCircle className="h-7 w-7 text-destructive/70" />
        </div>
        <div>
          <p className="text-sm font-semibold text-foreground">Camera Unavailable</p>
          <p className="text-xs text-muted-foreground mt-2 max-w-xs leading-relaxed">{errorMsg}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setPhase('idle')}>
            Back
          </Button>
          <Button size="sm" onClick={() => startCamera(facingMode)}>
            <RefreshCw className="h-3.5 w-3.5" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  /* ── Starting + Live ─────────────────────────────────────────── */
  return (
    <div className="space-y-3">
      {/* Video viewport */}
      <div className="relative overflow-hidden rounded-xl border border-border bg-black aspect-video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onCanPlay={() => setPhase('live')}
          className={cn(
            'w-full h-full object-cover transition-opacity duration-300',
            phase === 'live' ? 'opacity-100' : 'opacity-0',
          )}
        />

        {/* Viewfinder overlay (live only) */}
        {phase === 'live' && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-3 left-3 h-6 w-6 border-t-2 border-l-2 border-white/70 rounded-tl" />
            <div className="absolute top-3 right-3 h-6 w-6 border-t-2 border-r-2 border-white/70 rounded-tr" />
            <div className="absolute bottom-3 left-3 h-6 w-6 border-b-2 border-l-2 border-white/70 rounded-bl" />
            <div className="absolute bottom-3 right-3 h-6 w-6 border-b-2 border-r-2 border-white/70 rounded-br" />
            <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 rounded-full bg-black/50 px-2.5 py-1 backdrop-blur-sm">
              <motion.span
                animate={{ opacity: [1, 0.2, 1] }}
                transition={{ duration: 1.2, repeat: Infinity }}
                className="h-1.5 w-1.5 rounded-full bg-red-400"
              />
              <span className="text-[10px] font-bold tracking-widest text-white/80 uppercase">Live</span>
            </div>
          </div>
        )}

        {/* Starting spinner */}
        {phase === 'starting' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <motion.div
              animate={{ scale: [1, 1.15, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
              className="flex h-16 w-16 items-center justify-center rounded-full bg-white/10"
            >
              <Camera className="h-8 w-8 text-white/50" />
            </motion.div>
            <p className="text-xs text-white/40 font-mono">Starting camera…</p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        <Button onClick={capture} disabled={phase !== 'live' || disabled} className="flex-1" size="lg">
          <Camera className="h-4 w-4" />
          Capture Photo
        </Button>
        <Button variant="outline" size="lg" onClick={flipCamera} disabled={phase === 'starting'} title="Flip camera" aria-label="Flip camera">
          <FlipHorizontal className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="lg" onClick={() => { stopStream(); setPhase('idle'); }} title="Stop camera" aria-label="Stop camera">
          <VideoOff className="h-4 w-4" />
        </Button>
      </div>

      {/* Hidden canvas for snapshot */}
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
    </div>
  );
}

