import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useForm } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, X, Image as ImageIcon, FileWarning } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { validateImageFile, validateMimeType } from '@/utils/validators';
import { formatFileSize } from '@/utils/formatters';
import { cn } from '@/utils/cn';

/**
 * Drag-and-drop image upload component
 * Validates file type and size, shows preview
 */
export function ImageUpload({ onSubmit, isLoading }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const { handleSubmit } = useForm();

  const onDrop = useCallback(
    async (accepted) => {
      const img = accepted[0];
      if (!img) return;

      /* Basic validation */
      const { valid, error } = validateImageFile(img);
      if (!valid) {
        toast.error('Validation Error', { description: error });
        return;
      }

      /* MIME type deep check */
      const mimeOk = await validateMimeType(img);
      if (!mimeOk) {
        toast.error('Invalid File', {
          description: 'File header does not match a valid JPEG or PNG image.',
        });
        return;
      }

      setFile(img);
      setPreview(URL.createObjectURL(img));
    },
    [],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
    disabled: isLoading,
  });

  const clearFile = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
  };

  const onFormSubmit = () => {
    if (!file) {
      toast.error('No image', { description: 'Please upload an image first.' });
      return;
    }
    onSubmit(file);
  };

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-6 space-y-5">
        <h2 className="text-base font-semibold text-foreground">Upload Fundus Image</h2>

        <AnimatePresence mode="wait">
          {!preview ? (
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
                  isLoading && 'opacity-50 pointer-events-none',
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
                  <p className="text-xs text-muted-foreground mt-1">
                    JPG or PNG • Max 5MB
                  </p>
                </div>
              </div>
            </motion.div>
          ) : (
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
                {!isLoading && (
                  <button
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
                <span className="ml-auto">{formatFileSize(file?.size || 0)}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

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

        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-3">
          <FileWarning className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground leading-relaxed">
            Accepted formats: JPEG, PNG. Maximum file size: 5MB.
            Ensure the image is a clear retinal fundus photograph for best results.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
