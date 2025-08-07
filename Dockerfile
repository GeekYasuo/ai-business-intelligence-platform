# Multi-stage build for AI Business Intelligence Platform
# Optimized for production deployment with minimal footprint

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./

# Build optimized frontend
RUN npm run build

# Stage 2: Build backend
FROM node:18-alpine AS backend-builder

WORKDIR /app/backend

# Copy package files
COPY backend/package*.json ./
RUN npm ci --only=production

# Copy backend source
COPY backend/ ./

# Build TypeScript
RUN npm run build

# Stage 3: Python ML services
FROM python:3.9-slim AS ml-services

WORKDIR /app/ml-services

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY ml-services/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML service source
COPY ml-services/ ./

# Stage 4: Production runtime
FROM node:18-alpine AS production

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init python3 py3-pip

# Create app user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set working directory
WORKDIR /app

# Copy built backend from builder stage
COPY --from=backend-builder --chown=appuser:appgroup /app/backend/dist ./backend/dist
COPY --from=backend-builder --chown=appuser:appgroup /app/backend/node_modules ./backend/node_modules
COPY --from=backend-builder --chown=appuser:appgroup /app/backend/package*.json ./backend/

# Copy built frontend
COPY --from=frontend-builder --chown=appuser:appgroup /app/frontend/build ./frontend/build

# Copy ML services
COPY --from=ml-services --chown=appuser:appgroup /app/ml-services ./ml-services

# Create necessary directories
RUN mkdir -p logs uploads temp models && \
    chown -R appuser:appgroup logs uploads temp models

# Install Python dependencies for ML services
RUN pip3 install --no-cache-dir -r ml-services/requirements.txt

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 3001 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD node backend/dist/healthcheck.js || exit 1

# Set environment variables
ENV NODE_ENV=production \
    PORT=3001 \
    ML_SERVICE_PORT=8001 \
    LOG_LEVEL=info

# Start services with process manager
ENTRYPOINT ["dumb-init", "--"]
CMD ["sh", "-c", "python3 ml-services/main.py & node backend/dist/server.js"]
