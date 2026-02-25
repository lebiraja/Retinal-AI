import { Eye } from 'lucide-react';
import { Separator } from '@/components/ui/separator';

export function Footer() {
  return (
    <footer className="mt-auto border-t border-border/40">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 py-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Eye className="h-4 w-4" />
            <span className="text-sm font-medium">RetinalAI</span>
            <Separator orientation="vertical" className="h-4 mx-2" />
            <span className="text-xs">AI-Powered Retinal Disease Classifier</span>
          </div>
          <p className="text-xs text-muted-foreground/60">
            For screening purposes only — not a medical diagnostic device.
          </p>
        </div>
      </div>
    </footer>
  );
}
