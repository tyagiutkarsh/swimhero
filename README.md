# SwimHero - AI Swimming Analysis

AI-powered swimming technique analysis using Google Gemini Vision API.

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
./start.sh
```
The startup script will automatically:
- Create virtual environment if needed
- Install dependencies
- Check for API key configuration
- Start the server

### Option 2: Manual Setup
1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

4. **Run the server**:
   ```bash
   python main.py
   ```

5. **Open browser**: Navigate to `http://localhost:8000`

## Features

- **Drag-and-drop video upload** (MP4 files up to 200MB)
- **AI-powered analysis** using Gemini 2.5 Pro at 1 FPS
- **Interactive results** with video scrubbing and timeline navigation
- **Export options**: Copy, JSON, CSV formats
- **Stroke-specific analysis** (freestyle, backstroke, breaststroke, butterfly)
- **Camera perspective optimization** (side, underwater, above water)

## API

- `GET /` - Frontend application
- `POST /api/analyze` - Video analysis endpoint
- `GET /health` - Health check

## Requirements

- Python 3.8+
- Google AI API key ([Get one here](https://makersuite.google.com/app/apikey))
- MP4 video files (≤ 1 minute, ≤ 200MB)

## Tech Stack

- **Backend**: FastAPI + Google Generative AI
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **AI Model**: Gemini 2.5 Pro

For detailed development guidance, see [WARP.md](WARP.md).