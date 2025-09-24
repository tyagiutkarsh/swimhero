#!/bin/bash

# SwimHero Startup Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🏊‍♂️ SwimHero - Starting Server${NC}"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi, uvicorn, google.generativeai" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  Please edit .env and add your GOOGLE_API_KEY${NC}"
    echo -e "${RED}⚠️  Get your API key from: https://makersuite.google.com/app/apikey${NC}"
    exit 1
fi

# Check if API key is set
if grep -q "your_google_api_key_here" .env; then
    echo -e "${RED}⚠️  Please edit .env and add your GOOGLE_API_KEY${NC}"
    echo -e "${RED}⚠️  Get your API key from: https://makersuite.google.com/app/apikey${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment setup complete${NC}"
echo -e "${YELLOW}Starting SwimHero server...${NC}"
echo ""
echo -e "${GREEN}🌐 Server will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}📋 API documentation: http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python main.py