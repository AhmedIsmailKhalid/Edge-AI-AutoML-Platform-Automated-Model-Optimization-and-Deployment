#!/bin/bash
set -e

echo "Starting Edge AI AutoML Platform..."

# Create necessary directories and set permissions
mkdir -p /var/log/nginx /var/lib/nginx /run

# Test nginx configuration
echo "Testing nginx configuration..."
nginx -t

# Start nginx in background
echo "Starting nginx..."
nginx &
NGINX_PID=$!
sleep 2

# Check if nginx is actually running by testing port 7860
echo "Verifying nginx is listening on port 7860..."
for i in {1..10}; do
  if netstat -tuln 2>/dev/null | grep -q ":7860 " || ss -tuln 2>/dev/null | grep -q ":7860 "; then
    echo "Nginx is listening on port 7860!"
    break
  fi
  echo "Waiting for nginx... ($i/10)"
  sleep 1
done

# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
echo "Database URL: ${DATABASE_URL:-sqlite:///./experiments.db}"

uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "Backend is ready!"
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 2
done

# Final status check
echo "Checking service status..."
if ps -p $NGINX_PID > /dev/null 2>&1; then
  echo "✓ Nginx is running (PID: $NGINX_PID)"
else
  echo "✗ Nginx process not found"
fi

if ps -p $BACKEND_PID > /dev/null 2>&1; then
  echo "✓ Backend is running (PID: $BACKEND_PID)"
else
  echo "✗ Backend process not found"
fi

# Test if nginx is serving
echo "Testing nginx endpoint..."
if curl -s http://127.0.0.1:7860/health > /dev/null 2>&1; then
  echo "✓ Nginx is proxying requests successfully"
else
  echo "✗ Nginx proxy test failed"
  echo "Nginx error log:"
  cat /var/log/nginx/error.log 2>/dev/null || echo "No error log found"
fi

echo "All services started successfully!"
echo "Frontend: http://localhost:7860"
echo "Backend: http://localhost:7860/api"
echo "Docs: http://localhost:7860/docs"

# Keep container running by waiting for child processes
wait