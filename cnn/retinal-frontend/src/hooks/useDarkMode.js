import { useEffect } from 'react';
import useThemeStore from '@/store/useThemeStore';

/**
 * Hook to initialize and toggle dark mode
 * Reads persisted state from Zustand on mount
 */
export function useDarkMode() {
  const { isDark, toggleTheme, initTheme } = useThemeStore();

  useEffect(() => {
    initTheme();
  }, [initTheme]);

  return { isDark, toggleTheme };
}
