import os
import json
import tempfile
import logging
from functools import partial
import asyncio
import uuid
import time
from typing import Optional, List, Dict
from pathlib import Path

import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import ffmpeg

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swimhero.log')
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(title="SwimHero API", description="AI-powered swimming technique analysis")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Mount static files (serve the web frontend)
app.mount("/web", StaticFiles(directory="web"), name="web")

# In-memory storage for request tracking
active_requests: Dict[str, float] = {}

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=GOOGLE_API_KEY)

# Pydantic models
class FeedbackItem(BaseModel):
    timestamp: str
    issue: str
    suggestion: str

class AnalysisResponse(BaseModel):
    feedback: List[FeedbackItem]

# Constants
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
MAX_DURATION_SECONDS = 60  # 1 minute (trim to this)
TIMEOUT_SECONDS = 180  # 3 minutes
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming

def load_prompt_from_file() -> str:
    """Load the base prompt from prompt.md file."""
    try:
        prompt_path = Path("prompt.md")
        if not prompt_path.exists():
            raise FileNotFoundError("prompt.md file not found")
        
        content = prompt_path.read_text()
        
        return content
        
    except Exception as e:
        logger.error(f"Error loading prompt from file: {str(e)}")
        # Fallback to a minimal prompt if file loading fails
        return """Analyze this swimming video for technique issues and provide feedback in JSON format with timestamps, issues, and suggestions."""

def create_analysis_prompt(stroke: Optional[str] = None, camera_side: Optional[str] = None) -> str:
    """Create the analysis prompt for Gemini."""
    
    # Load base prompt from file
    base_prompt = load_prompt_from_file()
    
    # Add stroke-specific context
    if stroke:
        stroke_context = {
            "freestyle": "Focus on front crawl technique: rotation, high elbow catch, bilateral breathing.",
            "backstroke": "Focus on backstroke technique: body roll, straight arm recovery, consistent rhythm.",
            "breaststroke": "Focus on breaststroke technique: pull-breathe-kick-glide timing, streamline position.",
            "butterfly": "Focus on butterfly technique: undulation, simultaneous arm movement, dolphin kick timing."
        }
        if stroke in stroke_context:
            base_prompt += f"\n\nStroke-specific focus: {stroke_context[stroke]}"
    
    # Add camera position context
    if camera_side:
        camera_context = {
            "side": "Analyze from side view perspective: body position, stroke depth, kick effectiveness.",
            "underwater": "Analyze from underwater perspective: hand entry, catch phase, kick technique.",
            "above": "Analyze from above water perspective: arm recovery, breathing timing, stroke width."
        }
        if camera_side in camera_context:
            base_prompt += f"\n\nCamera perspective: {camera_context[camera_side]}"
    
    base_prompt += "\n\nReturn only valid JSON. No additional text or explanation."
    
    return base_prompt

def validate_and_sort_feedback(feedback_data: dict) -> List[FeedbackItem]:
    """Validate feedback data and sort by timestamp."""
    
    if not isinstance(feedback_data, dict) or "feedback" not in feedback_data:
        return []
    
    feedback_items = []
    
    for item in feedback_data["feedback"]:
        try:
            # Validate required fields
            if not all(key in item for key in ["timestamp", "issue", "suggestion"]):
                continue
            
            # Basic timestamp format validation
            timestamp = str(item["timestamp"])
            if ":" not in timestamp:
                continue
                
            feedback_items.append(FeedbackItem(
                timestamp=timestamp,
                issue=str(item["issue"]).strip(),
                suggestion=str(item["suggestion"]).strip()
            ))
        except Exception:
            # Skip invalid items
            continue
    
    # Sort by timestamp (basic string sort works for M:SS.s format)
    feedback_items.sort(key=lambda x: x.timestamp)
    
    # Remove near-duplicates (within ±3 second window)
    if len(feedback_items) <= 1:
        return feedback_items
    
    filtered_items = [feedback_items[0]]
    
    for item in feedback_items[1:]:
        # Parse timestamps for comparison
        try:
            current_seconds = parse_timestamp_to_seconds(item.timestamp)
            last_seconds = parse_timestamp_to_seconds(filtered_items[-1].timestamp)
            
            # If more than 3 seconds apart, keep it
            if abs(current_seconds - last_seconds) > 3:
                filtered_items.append(item)
        except Exception:
            # If parsing fails, keep the item
            filtered_items.append(item)
    
    return filtered_items

def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Parse M:SS.s timestamp to total seconds."""
    try:
        parts = timestamp.split(':')
        if len(parts) != 2:
            return 0
        
        minutes = int(parts[0])
        seconds_parts = parts[1].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) / 10 if len(seconds_parts) > 1 else 0
        
        return minutes * 60 + seconds + milliseconds
    except Exception:
        return 0

def trim_video_to_duration(input_path: str, output_path: str, duration_seconds: int = MAX_DURATION_SECONDS) -> bool:
    """Trim video to specified duration using ffmpeg."""
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, t=duration_seconds, c='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error during video trim: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during video trim: {str(e)}")
        return False

async def stream_upload_to_file(upload: UploadFile, temp_path: str) -> int:
    """Stream upload file to disk in chunks, return total bytes written."""
    total_size = 0
    
    with open(temp_path, 'wb') as temp_file:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
                
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                temp_file.close()
                os.unlink(temp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
                )
            
            temp_file.write(chunk)
    
    return total_size

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    try:
        frontend_path = Path("web/index.html")
        if not frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        
        return HTMLResponse(content=frontend_path.read_text(), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {str(e)}")

@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit("1/minute")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    stroke: Optional[str] = Form(None),
    camera_side: Optional[str] = Form(None)
):
    """Analyze swimming video using Gemini Vision."""
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Create logger with request ID context
    log_extra = {'request_id': request_id}
    req_logger = logging.LoggerAdapter(logger, log_extra)
    
    req_logger.info(f"Starting video analysis - File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # Validate file type
    if file.content_type != "video/mp4":
        req_logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail="Only MP4 video files are supported"
        )
    
    temp_file_path = None
    trimmed_file_path = None
    uploaded_file = None
    
    try:
        # Create temporary file paths
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)  # Close file descriptor, we'll use the path
        
        # Stream upload to disk with size validation
        req_logger.info("Starting file upload stream")
        file_size = await stream_upload_to_file(file, temp_file_path)
        req_logger.info(f"Upload complete - Size: {file_size} bytes")
        
        # Trim video to MAX_DURATION_SECONDS
        trimmed_fd, trimmed_file_path = tempfile.mkstemp(suffix="_trimmed.mp4")
        os.close(trimmed_fd)
        
        req_logger.info(f"Trimming video to {MAX_DURATION_SECONDS} seconds")
        if not trim_video_to_duration(temp_file_path, trimmed_file_path):
            raise HTTPException(status_code=500, detail="Video processing failed")
        
        # Upload to Gemini with timeout
        req_logger.info("Uploading to Gemini API")
        try:
            uploaded_file = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, genai.upload_file, trimmed_file_path
                ),
                timeout=TIMEOUT_SECONDS // 3  # Use 1/3 of total timeout for upload
            )
        except asyncio.TimeoutError:
            req_logger.error("Gemini upload timed out")
            raise HTTPException(status_code=504, detail="Video upload timeout")
        
        # Wait for processing with timeout
        req_logger.info("Waiting for Gemini processing")
        processing_start = time.time()
        try:
            while uploaded_file.state.name == "PROCESSING":
                if time.time() - processing_start > TIMEOUT_SECONDS // 2:
                    req_logger.error("Gemini processing timed out")
                    raise HTTPException(status_code=504, detail="Video processing timeout")
                    
                await asyncio.sleep(2)
                uploaded_file = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, genai.get_file, uploaded_file.name
                    ),
                    timeout=10
                )
        except asyncio.TimeoutError:
            req_logger.error("Gemini processing status check timed out")
            raise HTTPException(status_code=504, detail="Processing timeout")
        
        if uploaded_file.state.name == "FAILED":
            req_logger.error("Gemini video processing failed")
            raise HTTPException(status_code=500, detail="Video processing failed")
        
        # Create analysis prompt
        prompt = create_analysis_prompt(stroke, camera_side)
        
        # Initialize model and generate analysis
        model = genai.GenerativeModel(MODEL_NAME)
        
        req_logger.info("Generating AI analysis")
        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(
                        model.generate_content,
                        [uploaded_file, prompt],
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=4000,
                        ),
                    ),
                ),
                timeout=TIMEOUT_SECONDS // 3  # Use 1/3 of total timeout for generation
            )
        except asyncio.TimeoutError:
            req_logger.error("AI analysis generation timed out")
            raise HTTPException(status_code=504, detail="Analysis timeout")
        
        # Parse response
        response_text = response.text.strip()
        req_logger.info("Analysis complete, parsing results")
        
        # Try to extract JSON from response
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[start_idx:end_idx]
            feedback_data = json.loads(json_text)
            
        except (json.JSONDecodeError, ValueError) as e:
            req_logger.warning(f"JSON parsing failed: {str(e)}")
            feedback_data = {"feedback": []}
        
        # Validate and sort feedback
        validated_feedback = validate_and_sort_feedback(feedback_data)
        req_logger.info(f"Analysis complete - {len(validated_feedback)} feedback items")
        
        return AnalysisResponse(feedback=validated_feedback)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        req_logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed due to internal error"
        )
    finally:
        # Clean up temporary files
        cleanup_files = [temp_file_path, trimmed_file_path]
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    req_logger.info(f"Cleaned up temporary file")
                except Exception as cleanup_error:
                    req_logger.warning(f"Cleanup error: {cleanup_error}")
        
        # Clean up uploaded file
        if uploaded_file:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, genai.delete_file, uploaded_file.name
                )
                req_logger.info("Cleaned up Gemini uploaded file")
            except Exception as cleanup_error:
                req_logger.warning(f"Gemini cleanup error: {cleanup_error}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SwimHero API"}

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )