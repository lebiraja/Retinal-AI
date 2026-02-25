import { PageTransition } from '@/components/layout/PageTransition';
import { HistoryList } from '@/components/history/HistoryList';

export default function History() {
  return (
    <PageTransition>
      <div className="mx-auto max-w-3xl px-4 sm:px-6 py-10">
        <HistoryList />
      </div>
    </PageTransition>
  );
}
