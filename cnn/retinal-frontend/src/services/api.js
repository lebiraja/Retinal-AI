import axios from 'axios';

/**
 * Axios instance with base URL from environment
 * Centralized HTTP client for the entire app
 */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:7002',
  timeout: 30000,
  headers: {
    Accept: 'application/json',
  },
});

/* Request interceptor — log in dev */
api.interceptors.request.use(
  (config) => {
    if (import.meta.env.DEV) {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => Promise.reject(error),
);

/* Response interceptor — normalize errors */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';

    return Promise.reject(new Error(message));
  },
);

export default api;
