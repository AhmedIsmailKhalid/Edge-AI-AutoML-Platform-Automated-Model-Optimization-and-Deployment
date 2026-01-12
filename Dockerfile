FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (added net-tools for netstat)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    nginx \
    net-tools \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/pretrained/ ./models/pretrained/
COPY dataset/preset/ ./dataset/preset/

# Copy frontend and build
COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Verify build succeeded
RUN ls -la /app/frontend/dist/ && \
    test -f /app/frontend/dist/index.html || (echo "Frontend build failed!" && exit 1)

# Back to app root
WORKDIR /app

# Create runtime directories with proper permissions
RUN mkdir -p models/custom dataset/custom outputs \
    /var/log/nginx /var/lib/nginx /var/lib/nginx/body /var/lib/nginx/proxy \
    /run && \
    chmod -R 777 /var/log/nginx /var/lib/nginx /run

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy and setup start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && \
    sed -i 's/\r$//' /app/start.sh

# Environment variables
ENV DATABASE_URL=sqlite:///./experiments.db \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start services
CMD ["/bin/bash", "/app/start.sh"]