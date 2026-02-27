import { useCallback, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useForm } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, X, Image as ImageIcon, FileWarning, Camera } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { validateImageFile, validateMimeType } from '@/utils/validators';
import { formatFileSize } from '@/utils/formatters';
import { cn } from '@/utils/cn';
import { ScanOverlay } from './ScanOverlay';
import { CameraCapture } from './CameraCapture';

const TABS = [
  { id: 'upload', label: 'Upload',  Icon: UploadCloud },
  { id: 'camera', label: 'Camera',  Icon: Camera },
];

/**
 * Fundus image input panel.
 *
 * - Upload tab : drag-and-drop / file browser
 * - Camera tab : live webcam feed with snapshot capture
 * - While isLoading : retinal scan overlay animates over the preview
 */
export function ImageUpload({ onSubmit, isLoading }) {
  const [tab, setTab]       = useState('upload'); // 'upload' | 'camera'
  const [preview, setPreview] = useState(null);
  const [file, setFile]     = useState(null);
  const previewUrlRef       = useRef(null);
  const { handleSubmit }    = useForm();

  /* ── Shared helper: store a new image ─────────────────────────── */
  const setImage = useCallback((newFile, objectUrl) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = objectUrl;
    setFile(newFile);
    setPreview(objectUrl);
  }, []);

  /* ── Upload tab: dropzone ─────────────────────────────────────── */
  const onDrop = useCallback(
    async (accepted) => {
      const img = accepted[0];
      if (!img) return;

      const { valid, error } = validateImageFile(img);
      if (!valid) { toast.error('Validation Error', { description: error }); return; }

      const mimeOk = await validateMimeType(img);
      if (!mimeOk) {
        toast.error('Invalid File', {
          description: 'File header does not match a valid JPEG or PNG image.',
        });
        return;
      }

      setImage(img, URL.createObjectURL(img));
    },
    [setImage],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
    disabled: isLoading,
  });

  /* ── Camera tab: capture callback ─────────────────────────────── */
  const handleCapture = useCallback(
    (capturedFile) => {
      setImage(capturedFile, URL.createObjectURL(capturedFile));
      setTab('upload'); // switch to upload tab to show preview + submit
    },
    [setImage],
  );

  /* ── Clear image ──────────────────────────────────────────────── */
  const clearFile = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setFile(null);
    setPreview(null);
  };

  /* ── Submit ───────────────────────────────────────────────────── */
  const onFormSubmit = () => {
    if (!file) {
      toast.error('No image', { description: 'Please upload or capture an image first.' });
      return;
    }
    onSubmit(file);
  };

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-6 space-y-5">
        <h2 className="text-base font-semibold text-foreground">Fundus Image</h2>

        {/* ── Tab bar ──────────────────────────────────────────────── */}
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              disabled={isLoading}
              className={cn(
                'flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-sm font-medium transition-all',
                tab === id
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
                isLoading && 'pointer-events-none opacity-50',
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* ── Tab content ──────────────────────────────────────────── */}
        <AnimatePresence mode="wait">

          {/* Upload tab */}
          {tab === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <AnimatePresence mode="wait">
                {!preview ? (
                  /* Dropzone */
                  <motion.div
                    key="dropzone"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <div
                      {...getRootProps()}
                      className={cn(
                        'flex flex-col items-center gap-3 rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-200',
                        isDragActive
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50 hover:bg-accent/50',
                        isLoading && 'pointer-events-none opacity-50',
                      )}
                    >
                      <input {...getInputProps()} />
                      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                        <UploadCloud className="h-7 w-7 text-primary" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-foreground">
                          {isDragActive ? 'Drop image here' : 'Drag & drop or click to browse'}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">JPG or PNG • Max 5 MB</p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  /* Preview + scan overlay */
                  <motion.div
                    key="preview"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="relative"
                  >
                    <div className="relative overflow-hidden rounded-xl border border-border">
                      <img
                        src={preview}
                        alt="Fundus preview"
                        className="w-full h-56 object-cover"
                      />

                      {/* Retinal scan animation while analyzing */}
                      <AnimatePresence>
                        {isLoading && (
                          <motion.div
                            key="scan-overlay"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0"
                          >
                            <ScanOverlay />
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {/* Remove button (hidden while loading) */}
                      {!isLoading && (
                        <button
                          type="button"
                          onClick={clearFile}
                          className="absolute top-2 right-2 flex h-8 w-8 items-center justify-center rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-sm hover:bg-destructive hover:text-destructive-foreground transition-colors"
                          aria-label="Remove image"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>

                    <div className="flex items-center gap-2 mt-2.5 text-xs text-muted-foreground">
                      <ImageIcon className="h-3.5 w-3.5" />
                      <span className="truncate">{file?.name}</span>
                      <span className="ml-auto shrink-0">{formatFileSize(file?.size || 0)}</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {/* Camera tab */}
          {tab === 'camera' && (
            <motion.div
              key="camera"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <CameraCapture onCapture={handleCapture} disabled={isLoading} />
            </motion.div>
          )}

        </AnimatePresence>

        {/* ── Submit button (upload tab only) ──────────────────────── */}
        {tab === 'upload' && (
          <form onSubmit={handleSubmit(onFormSubmit)}>
            <Button
              type="submit"
              className="w-full"
              size="lg"
              disabled={!file || isLoading}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
                  Analyzing…
                </span>
              ) : (
                'Start Analysis'
              )}
            </Button>
          </form>
        )}

        {/* ── Hint ─────────────────────────────────────────────────── */}
        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3">
          <FileWarning className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground leading-relaxed">
            {tab === 'camera'
              ? 'Position the retinal fundus image in front of the camera and press Capture Photo.'
              : 'Accepted formats: JPEG, PNG. Maximum file size: 5 MB. Ensure the image is a clear retinal fundus photograph for best results.'}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
