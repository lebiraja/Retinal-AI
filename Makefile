# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — Developer shortcuts for the Retinal Disease Classifier stack
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help build up down restart logs clean ps \
        build-model build-backend build-frontend build-nginx \
        up-cpu up-cpu-d \
        shell-model shell-backend \
        gpu-check lint test

COMPOSE        := docker compose
COMPOSE_PROD   := $(COMPOSE) -f docker-compose.yml
COMPOSE_CPU    := $(COMPOSE) -f docker-compose.yml -f docker-compose.cpu.yml
GPU_ID         ?= 0

# ── Default target ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Retinal Disease Classifier — Docker Makefile"
	@echo ""
	@echo "  make build        ── Build all service images"
	@echo "  make up           ── Start all services (dev mode with override)"
	@echo "  make up-prod      ── Start all services (production, no override)"
	@echo "  make down         ── Stop and remove containers"
	@echo "  make restart      ── Rebuild and restart all services"
	@echo "  make logs         ── Tail logs for all services"
	@echo "  make logs-model   ── Tail model service logs"
	@echo "  make logs-backend ── Tail backend service logs"
	@echo "  make ps           ── Show running containers"
	@echo "  make gpu-check    ── Verify NVIDIA GPU is visible inside containers"
	@echo "  make clean        ── Remove images and volumes (DESTRUCTIVE)"
	@echo "  make test         ── Run backend unit tests"
	@echo ""

# ── Build ──────────────────────────────────────────────────────────────────────
build:
	$(COMPOSE) build

build-model:
	$(COMPOSE) build model-service

build-backend:
	$(COMPOSE) build backend

build-frontend:
	$(COMPOSE) build frontend

build-nginx:
	$(COMPOSE) build nginx

# ── Up / Down ──────────────────────────────────────────────────────────────────
up:
	$(COMPOSE) up --build

up-d:
	$(COMPOSE) up --build -d

up-prod:
	$(COMPOSE_PROD) up --build -d

## CPU-only (no GPU required) — for testing on any machine
up-cpu:
	$(COMPOSE_CPU) up --build

up-cpu-d:
	$(COMPOSE_CPU) up --build -d

down:
	$(COMPOSE) down

restart: down build up-d

# ── Logs ───────────────────────────────────────────────────────────────────────
logs:
	$(COMPOSE) logs -f

logs-model:
	$(COMPOSE) logs -f model-service

logs-backend:
	$(COMPOSE) logs -f backend

logs-frontend:
	$(COMPOSE) logs -f frontend

logs-nginx:
	$(COMPOSE) logs -f nginx

# ── Status ─────────────────────────────────────────────────────────────────────
ps:
	$(COMPOSE) ps

# ── Shell access ───────────────────────────────────────────────────────────────
shell-model:
	$(COMPOSE) exec model-service /bin/bash

shell-backend:
	$(COMPOSE) exec backend /bin/bash

# ── GPU verification ───────────────────────────────────────────────────────────
gpu-check:
	@echo "==> Host NVIDIA driver"
	@nvidia-smi || echo "nvidia-smi not found on host"
	@echo ""
	@echo "==> GPU visibility inside model-service container"
	$(COMPOSE) exec model-service python -c \
		"import torch; print('CUDA available:', torch.cuda.is_available()); \
		 [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

# ── Health checks ──────────────────────────────────────────────────────────────
health:
	@echo "==> nginx"
	@curl -s http://localhost/nginx-health
	@echo ""
	@echo "==> backend (via nginx)"
	@curl -s http://localhost/api/health | python3 -m json.tool
	@echo ""
	@echo "==> model service (direct)"
	@curl -s http://localhost:8001/health | python3 -m json.tool || \
		echo "(port 8001 not exposed in production mode)"

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	$(COMPOSE) exec backend python -m pytest tests/ -v

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	@echo "WARNING: This will remove all retinal project images and the HF model cache."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) down -v --rmi all
