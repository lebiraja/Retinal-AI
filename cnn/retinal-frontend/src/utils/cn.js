import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge Tailwind classes with clsx + tailwind-merge
 * Prevents duplicate/conflicting utility classes
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
