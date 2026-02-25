import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Theme store — controls dark/light mode
 * Persists user preference in localStorage
 */
const useThemeStore = create(
  persist(
    (set, get) => ({
      isDark: false,

      /** Toggle dark mode */
      toggleTheme: () => {
        const next = !get().isDark;
        set({ isDark: next });
        applyTheme(next);
      },

      /** Initialize theme from persisted state on app mount */
      initTheme: () => {
        applyTheme(get().isDark);
      },
    }),
    {
      name: 'retinal-theme',
    },
  ),
);

/** Apply the dark class to the document root */
function applyTheme(isDark) {
  const root = document.documentElement;
  if (isDark) {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
}

export default useThemeStore;
