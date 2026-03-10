# ─────────────────────────────────────────────────────────────────────────────
#  Nginx Reverse Proxy — Docker image
#
#  This is the single public entry point for the entire stack.
#  It routes:
#    /api/*  → backend service
#    /*      → frontend service
# ─────────────────────────────────────────────────────────────────────────────

FROM nginx:1.27-alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our reverse proxy config
COPY nginx/nginx.conf /etc/nginx/conf.d/proxy.conf

# Basic 50x error page
RUN mkdir -p /usr/share/nginx/html && \
    printf '<!doctype html><html><head><title>Service Unavailable</title></head>' \
           '<body><h2>503 — Service temporarily unavailable.</h2>' \
           '<p>The server is starting up. Please try again in a moment.</p>' \
           '</body></html>' > /usr/share/nginx/html/50x.html

RUN nginx -t

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q --spider http://localhost/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]
