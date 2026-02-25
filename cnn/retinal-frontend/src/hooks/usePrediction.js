import { useMutation } from '@tanstack/react-query';
import { toast } from 'sonner';
import { predictImage } from '@/services/prediction';
import useHistoryStore from '@/store/useHistoryStore';

/**
 * Custom hook for image prediction
 * Wraps TanStack Query mutation with history storage + toast
 */
export function usePrediction() {
  const addPrediction = useHistoryStore((s) => s.addPrediction);

  const mutation = useMutation({
    mutationFn: predictImage,
    onSuccess: (data) => {
      addPrediction(data);
      const topName = data.predictions?.[0]?.fullName || data.topPrediction || 'None';
      toast.success('Analysis complete', {
        description: `Top finding: ${topName}`,
      });
    },
    onError: (error) => {
      toast.error('Analysis failed', {
        description: error.message || 'Please try again.',
      });
    },
  });

  return {
    predict: mutation.mutate,
    predictAsync: mutation.mutateAsync,
    data: mutation.data,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
    reset: mutation.reset,
  };
}
