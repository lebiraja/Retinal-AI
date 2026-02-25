import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const MAX_HISTORY = 5;

/**
 * Prediction history store
 * Persists last 5 predictions in localStorage
 */
const useHistoryStore = create(
  persist(
    (set, get) => ({
      history: [],

      /** Add a prediction result to history (FIFO, max 5) */
      addPrediction: (result) => {
        const top = result.predictions?.[0];
        const entry = {
          id: Date.now(),
          disease: top?.fullName || top?.disease || result.topPrediction || 'Unknown',
          confidence: top?.confidence ?? result.confidence ?? 0,
          riskLevel: result.riskLevel || 'UNKNOWN',
          numDetected: result.numDetected || 0,
          allPredictions: result.predictions || [],
          processingTime: result.processing_time,
          timestamp: new Date().toISOString(),
        };
        set((state) => ({
          history: [entry, ...state.history].slice(0, MAX_HISTORY),
        }));
      },

      /** Clear all history */
      clearHistory: () => set({ history: [] }),

      /** Remove a single entry by id */
      removeEntry: (id) =>
        set((state) => ({
          history: state.history.filter((h) => h.id !== id),
        })),
    }),
    {
      name: 'retinal-history',
    },
  ),
);

export default useHistoryStore;
