# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
python3 -m venv venv               # Create virtual environment
source venv/bin/activate           # Activate virtual environment
pip install -r requirements.txt    # Install Python dependencies
cp .env.example .env               # Copy environment template
# Edit .env with your GOOGLE_API_KEY
```

### Quick Start (Automated)
```bash
./start.sh                         # Run automated setup script
```

### Docker
```bash
docker build -t swimhero .
docker run -p 8000:8000 -e GOOGLE_API_KEY={gemini_api_key} swimhero
```

### Development Server
```bash
source venv/bin/activate           # Always activate virtual environment first
python main.py                     # Start development server (auto-reload)
uvicorn main:app --reload          # Alternative way to start dev server
uvicorn main:app --host 0.0.0.0 --port 8000  # Start with specific host/port
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4  # Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload=false  # Production mode
```

### Code Quality
```bash
black .                           # Format Python code
isort .                           # Sort imports
flake8 .                          # Lint Python code
mypy main.py                      # Type checking
```

### Testing and Debugging
```bash
curl -X GET http://localhost:8000/health          # Health check
curl -X POST http://localhost:8000/api/analyze \   # Test API endpoint (max 200MB, 1min)
  -F "file=@test_video.mp4" \
  -F "stroke=freestyle" \
  -F "camera_side=side"
python -c "import google.generativeai; print('Gemini SDK OK')"  # Test Gemini SDK
tail -f swimhero.log                              # Monitor logs
ffmpeg -i input.mp4 -t 60 -c copy output.mp4     # Test video trimming
```

## Project Architecture

### Core Structure
- **Frontend**: Single-page vanilla JavaScript application (`web/index.html`)
- **Backend**: FastAPI server (`main.py`) with Gemini AI integration
- **AI Model**: Google Gemini 2.5 Pro for video analysis
- **File Storage**: Temporary file handling for video processing

### Key Files
- `main.py` - FastAPI backend with `/api/analyze` endpoint and static serving
- `web/index.html` - Complete frontend (HTML + CSS + JavaScript)
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `WARP.md` - This guidance file

### Request Flow
1. **Upload**: Frontend validates MP4 file (≤500MB, ≤5min) via drag-drop
2. **Processing**: Backend saves temp file, uploads to Gemini API
3. **Analysis**: Gemini processes video at 1 FPS with structured prompt
4. **Response**: JSON feedback with timestamps, issues, and suggestions
5. **Display**: Frontend shows results table with video scrubbing

### Data Models (Swimming Context)
- **Users**: Swimmers, coaches, administrators
- **Workouts**: Training sessions, sets, intervals
- **Performance**: Times, distances, stroke data
- **Goals**: Training objectives and achievements
- **Analytics**: Performance tracking and insights

## Development Guidelines

### TypeScript Preferences
- Use `type` instead of `interface` for type definitions
- Maintain strict type checking
- Export types from dedicated type files

### Code Style
- Follow Hemingway Test principles for JSDoc comments (concise, clear)
- Use descriptive variable and function names
- Prefer functional programming patterns where appropriate

### Testing Strategy
- Write unit tests for all utilities and business logic
- Integration tests for API endpoints
- E2E tests for critical user flows
- Maintain good test coverage

### Performance Considerations
- Optimize for swimming data visualization (charts, graphs)
- Handle large datasets efficiently (workout history, analytics)
- Implement proper caching strategies

## Frontend UX Implementation

### Upload Flow States
1. **Initial**: Drag-drop zone with file validation (MP4, ≤200MB)
2. **File Selected**: Shows filename, size, stroke/camera options
3. **Progress**: Queued → Thinking → Finalizing with consolidated progress bar
4. **Results**: Video player + feedback table with clickable timestamps
5. **Export**: Copy, JSON export, CSV download, "Analyze Another" reset

### Key UX Behaviors
- **Drag-drop validation**: Real-time file type and size checking (200MB limit)
- **Streaming upload**: Chunked file transfer to avoid memory issues
- **Progress mapping**: Upload (10%) → API call (50%) → Results (90%) → Complete (100%)
- **Video scrubbing**: Click table row → parse timestamp → `video.currentTime` sync
- **Export functions**: Clipboard API, blob downloads, proper filename generation
- **Error handling**: Toast notifications for validation, network, timeout, and API errors
- **Rate limiting**: 1 request per minute per IP address

## Backend Technical Details

### Gemini Integration
- **Model**: `gemini-2.5-pro` (configurable via `MODEL_NAME`)
- **Processing**: 1 FPS sampling with structured JSON prompt
- **Video trimming**: Automatically trim to first 60 seconds using FFmpeg
- **Temperature**: 0.3 for consistent technical analysis
- **Timeout**: 180s total (60s upload + 90s processing + 30s generation)
- **Streaming upload**: Chunked file transfer (1MB chunks)
- **Rate limiting**: 1 request/minute per IP using SlowAPI
- **Cleanup**: Automatic temp file and uploaded file deletion

### API Response Format
```json
{
  "feedback": [
    {
      "timestamp": "M:SS.s",
      "issue": "Brief technique issue description",
      "suggestion": "Specific improvement recommendation"
    }
  ]
}
```

### Data Processing
- **Validation**: Required fields, timestamp format checking
- **Sorting**: Chronological order by timestamp
- **Deduplication**: ±3 second window for near-duplicate removal
- **Error fallback**: Empty feedback array on parsing failures

### Logging and Monitoring
- **Structured logging**: Request IDs, timestamps, log levels
- **Log file**: `swimhero.log` with rotation
- **Request tracking**: Each request gets unique 8-char ID
- **Performance metrics**: Upload size, processing time tracking
- **Error tracking**: Categorized by type (validation, timeout, API, parsing)
- **Security**: Sensitive data (filenames, IPs) redacted in logs

## Swimming Analysis Context

### Supported Strokes
- **Freestyle**: Front crawl technique, rotation, high elbow catch
- **Backstroke**: Body roll, straight arm recovery, rhythm
- **Breaststroke**: Pull-breathe-kick-glide timing, streamline
- **Butterfly**: Undulation, simultaneous movement, dolphin kick

### Camera Perspectives
- **Side view**: Body position, stroke depth, kick effectiveness
- **Underwater**: Hand entry, catch phase, kick technique  
- **Above water**: Arm recovery, breathing timing, stroke width

### Analysis Focus Areas
- Body position and alignment
- Stroke mechanics (catch, pull, recovery)
- Breathing technique and timing
- Kick effectiveness and rhythm
- Entry and exit technique
- Overall coordination and efficiency

## Environment Setup

### Prerequisites
- Python 3.8+ (recommended: Python 3.11)
- pip package manager
- Google AI API key (from Google AI Studio)

### Configuration
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Environment setup**: `cp .env.example .env`
3. **API Key**: Add your `GOOGLE_API_KEY` to `.env` file
4. **Model**: Optionally specify `MODEL_NAME` (defaults to `gemini-2.5-pro`)

### API Key Setup
- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Add to `.env` as `GOOGLE_API_KEY=your_key_here`
- Never commit API keys to version control

## Deployment

### Build Process
- Run type checking before deployment
- Format code before commits
- Ensure all tests pass
- Build optimized production bundle

### Staging/Production
- Follow CI/CD pipeline defined in repository
- Database migrations handled automatically or documented
- Environment-specific configuration management