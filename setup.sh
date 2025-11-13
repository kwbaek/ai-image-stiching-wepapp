#!/bin/bash

# AI Image Stitching WebApp Setup Script for macOS
# This script sets up both backend and frontend

set -e  # Exit on error

echo "🚀 AI Image Stitching WebApp Setup"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

echo "✅ Python version: $(python3 --version)"
echo "✅ Node.js version: $(node --version)"
echo ""

# Backend Setup
echo "📦 Setting up Backend..."
echo "------------------------"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

# Frontend Setup
cd ../frontend
echo "📦 Setting up Frontend..."
echo "------------------------"

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"
echo ""

# Done
cd ..
echo "✨ Setup Complete!"
echo ""
echo "To run the application:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload --port 8000"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser!"
echo ""
echo "🤖 Using LoFTR Transformer model for AI-powered image stitching"
echo "   GPU acceleration will be used if CUDA is available"
