/**
 * Format confidence score as percentage string
 * @param {number} value - 0..1
 * @returns {string}
 */
export function formatConfidence(value) {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format processing time
 * @param {number} seconds
 * @returns {string}
 */
export function formatProcessingTime(seconds) {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(2)}s`;
}

/**
 * Format file size in human-readable form
 * @param {number} bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Format a Date object to a readable timestamp
 * @param {Date|string} date
 * @returns {string}
 */
export function formatTimestamp(date) {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Get risk level based on top confidence
 * @param {number} confidence - 0..1
 * @returns {{ level: string, color: string }}
 */
export function getRiskLevel(confidence) {
  if (confidence >= 0.8) return { level: 'High', color: 'destructive' };
  if (confidence >= 0.5) return { level: 'Moderate', color: 'warning' };
  return { level: 'Low', color: 'success' };
}
