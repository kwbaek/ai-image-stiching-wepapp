#!/bin/bash

# Run Backend Server

cd backend

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting Backend Server..."
echo "API will be available at: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

# Run FastAPI server
uvicorn app.main:app --reload --port 8000
