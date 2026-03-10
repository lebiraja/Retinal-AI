# ─────────────────────────────────────────────────────────────────────────────
#  Frontend Service — React/Vite app served by Nginx
#
#  Stage 1: Build the React app (node:20-alpine)
#  Stage 2: Serve static files (nginx:1.27-alpine)
#
#  Build:
#    docker build \
#      --build-arg VITE_API_URL=/api \
#      -t retinal/frontend:latest \
#      -f services/frontend/Dockerfile \
#      .
#
#  The build context MUST be the project root so we can COPY the frontend src.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Node.js build ─────────────────────────────────────────────────────
FROM node:20-alpine AS builder

# Build-time environment variable — the URL the browser will use to reach the API.
# In production through Nginx: /api  (nginx proxies /api/ → backend:8000)
# In local dev override:       http://localhost:8000
ARG VITE_API_URL=/api

ENV VITE_API_URL=${VITE_API_URL}

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY cnn/retinal-frontend/package.json cnn/retinal-frontend/package-lock.json* ./

# Use npm ci for reproducible installs (preferred over npm install in CI/Docker)
RUN npm ci --prefer-offline

# Copy the full source tree
COPY cnn/retinal-frontend/ ./

# Production build
RUN npm run build


# ── Stage 2: Nginx production server ──────────────────────────────────────────
FROM nginx:1.27-alpine AS production

# Remove the default Nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom config
COPY services/frontend/nginx.conf /etc/nginx/conf.d/app.conf

# Copy the built static files from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx runs as root by default; keep it that way inside the container since
# it needs port 80. The container itself is isolated by Docker networking.

EXPOSE 80

# Verify config before starting
RUN nginx -t

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
