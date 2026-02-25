import { Outlet } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Footer } from './Footer';

/**
 * Root layout shell — wraps all routed pages
 */
export function Layout() {
  return (
    <div className="relative flex min-h-screen flex-col">
      {/* Subtle background gradient */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-gradient-to-br from-primary/[0.03] via-transparent to-primary/[0.02]" />
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
