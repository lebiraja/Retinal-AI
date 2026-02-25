import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { AnimatePresence } from 'framer-motion';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Layout } from '@/components/layout/Layout';
import { Skeleton } from '@/components/ui/skeleton';

/* Lazy-loaded pages */
const Landing = lazy(() => import('@/pages/Landing'));
const Predict = lazy(() => import('@/pages/Predict'));
const History = lazy(() => import('@/pages/History'));
const HowItWorks = lazy(() => import('@/pages/HowItWorks'));
const NotFound = lazy(() => import('@/pages/NotFound'));

/* TanStack Query client */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 1000 * 60 * 5,
      refetchOnWindowFocus: false,
    },
  },
});

/* Suspense fallback */
function PageLoader() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10 space-y-4">
      <Skeleton className="h-10 w-64 rounded-xl" />
      <Skeleton className="h-4 w-96 rounded-lg" />
      <div className="grid lg:grid-cols-5 gap-6 mt-8">
        <Skeleton className="lg:col-span-2 h-80 rounded-2xl" />
        <Skeleton className="lg:col-span-3 h-80 rounded-2xl" />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <BrowserRouter>
          <Toaster
            position="top-right"
            richColors
            closeButton
            toastOptions={{
              className: 'font-sans',
            }}
          />
          <AnimatePresence mode="wait">
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route element={<Layout />}>
                  <Route index element={<Landing />} />
                  <Route path="predict" element={<Predict />} />
                  <Route path="history" element={<History />} />
                  <Route path="how-it-works" element={<HowItWorks />} />
                  <Route path="*" element={<NotFound />} />
                </Route>
              </Routes>
            </Suspense>
          </AnimatePresence>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
