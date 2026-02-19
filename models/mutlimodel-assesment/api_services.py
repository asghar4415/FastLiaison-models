"""
FastAPI Service for Video Emotion Recognition and Transcription
Processes uploaded MP4 videos and returns comprehensive emotion insights
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import os
import wave
from typing import Dict, List, Optional
from datetime import datetime
import torch
from collections import defaultdict, Counter
import whisper
from pydantic import BaseModel
import logging

# Directory of this script, used for temp files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import local utilities
from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor
from config.config import EMOTION_LABELS


# Pydantic models for responses
class EmotionFrame(BaseModel):
    timestamp: float
    frame_number: int
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    faces_detected: int


class EmotionStats(BaseModel):
    emotion: str
    count: int
    percentage: float
    avg_confidence: float


class VideoInsights(BaseModel):
    video_metadata: Dict
    emotion_analysis: Dict
    transcription: Optional[Dict] = None
    processing_time: float


# Initialize FastAPI app
app = FastAPI(
    title="MMIA Video Emotion Recognition API",
    description="Analyze emotions in video files and transcribe audio",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loading)
face_detector = None
emotion_predictor = None
whisper_model = None


def get_face_detector():
    """Lazy load face detector"""
    global face_detector
    if face_detector is None:
        logger.info("Loading face detector...")
        face_detector = FaceDetector()
    return face_detector


def get_emotion_predictor():
    """Lazy load emotion predictor"""
    global emotion_predictor
    if emotion_predictor is None:
        logger.info("Loading emotion predictor...")
        emotion_predictor = EmotionPredictor()
    return emotion_predictor


def get_whisper_model():
    """Lazy load Whisper model for transcription"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        # Use base model for balance of speed and accuracy
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded")
    return whisper_model


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using moviepy 1.0.3
    
    Args:
        video_path: Path to video file
        audio_path: Path to save extracted audio
        
    Returns:
        True if audio extracted successfully, False otherwise
    """
    try:
        from moviepy.editor import VideoFileClip
        
        logger.info(f"Loading video for audio extraction: {video_path}")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            logger.warning("No audio stream found in video")
            video.close()
            return False
        
        logger.info(f"Extracting audio to: {audio_path}")
        # Write audio to file - moviepy 1.0.3 compatible
        video.audio.write_audiofile(
            audio_path,
            fps=16000,  # Sample rate
            nbytes=2,   # 16-bit
            codec='pcm_s16le',
            verbose=False,
            logger=None  # Suppress moviepy output
        )
        
        video.close()
        
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        if success:
            logger.info(f"Audio extracted successfully: {os.path.getsize(audio_path)} bytes")
        else:
            logger.warning("Audio extraction failed or resulted in empty file")
        
        return success
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}", exc_info=True)
        return False


def load_wav_as_numpy(wav_path: str) -> np.ndarray:
    """
    Load a PCM WAV file as a float32 numpy array suitable for Whisper.
    This avoids Whisper's internal load_audio which requires system ffmpeg.
    
    Args:
        wav_path: Path to WAV file (PCM 16-bit, 16kHz expected)
        
    Returns:
        Mono float32 numpy array normalized to [-1, 1]
    """
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sample_width == 2:  # 16-bit
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:  # 32-bit
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    # Convert to mono if multi-channel
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio


def transcribe_audio(audio_path: str) -> Dict:
    """
    Transcribe audio using Whisper
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with transcription results
    """
    try:
        model = get_whisper_model()
        
        logger.info(f"Starting transcription of: {audio_path}")
        
        # Load WAV as numpy array to avoid Whisper calling system ffmpeg
        audio_array = load_wav_as_numpy(audio_path)
        logger.info(f"Loaded audio: {len(audio_array)} samples ({len(audio_array)/16000:.1f}s)")
        
        result = model.transcribe(
            audio_array,
            language=None,  # Auto-detect language
            task="transcribe",
            fp16=False,  # Disable FP16 for CPU compatibility
            verbose=False
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': round(segment['start'], 2),
                'end': round(segment['end'], 2),
                'text': segment['text'].strip(),
                'confidence': round(1.0 - segment.get('no_speech_prob', 0.0), 4)
            })
        
        full_text = result.get('text', '').strip()
        language = result.get('language', 'unknown')
        
        logger.info(f"Transcription complete: {len(segments)} segments, language: {language}")
        if full_text:
            logger.info(f"Transcript preview: {full_text[:100]}...")
        else:
            logger.warning("Transcription resulted in empty text (possibly no speech in audio)")
        
        return {
            'success': True,
            'full_text': full_text,
            'language': language,
            'segments': segments,
            'segment_count': len(segments)
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'full_text': '',
            'segments': [],
            'segment_count': 0
        }


def analyze_video_emotions(video_path: str, sample_rate: int = 1) -> Dict:
    """
    Analyze emotions in video file frame by frame
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame (1 = every frame)
        
    Returns:
        Dictionary with comprehensive emotion analysis
    """
    detector = get_face_detector()
    predictor = get_emotion_predictor()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Analyzing video: {total_frames} frames, {fps} fps, {duration:.2f}s")
    
    # Storage for analysis
    frame_emotions = []
    emotion_counts = defaultdict(int)
    emotion_confidences = defaultdict(list)
    frames_with_faces = 0
    frames_without_faces = 0
    
    frame_number = 0
    processed_frames = 0
    
    # Reset LSTM buffer at start
    predictor.reset_lstm_buffer()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every Nth frame
            if frame_number % sample_rate == 0:
                timestamp = frame_number / fps if fps > 0 else 0
                
                # Detect faces
                faces = detector.detect_faces(frame)
                
                if len(faces) > 0:
                    frames_with_faces += 1
                    
                    # Process first detected face
                    bbox = faces[0]
                    face_img = detector.extract_face(frame, bbox)
                    
                    # Predict emotion
                    emotion, confidence, probabilities = predictor.predict_emotion(face_img)
                    
                    # Store results
                    frame_emotions.append({
                        'timestamp': round(timestamp, 2),
                        'frame_number': frame_number,
                        'emotion': emotion,
                        'confidence': round(float(confidence), 4),
                        'all_probabilities': {
                            k: round(float(v), 4) for k, v in probabilities.items()
                        },
                        'faces_detected': len(faces)
                    })
                    
                    # Update statistics
                    emotion_counts[emotion] += 1
                    emotion_confidences[emotion].append(confidence)
                    
                else:
                    frames_without_faces += 1
                    # Add frame with no detection
                    frame_emotions.append({
                        'timestamp': round(timestamp, 2),
                        'frame_number': frame_number,
                        'emotion': 'No Face Detected',
                        'confidence': 0.0,
                        'all_probabilities': {},
                        'faces_detected': 0
                    })
                
                processed_frames += 1
            
            frame_number += 1
            
    finally:
        cap.release()
    
    logger.info(f"Processed {processed_frames} frames, {frames_with_faces} with faces")
    
    # Calculate statistics
    total_detected_frames = sum(emotion_counts.values())
    
    emotion_statistics = []
    for emotion in EMOTION_LABELS.values():
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total_detected_frames * 100) if total_detected_frames > 0 else 0
        avg_conf = np.mean(emotion_confidences.get(emotion, [0])) if emotion in emotion_confidences else 0
        
        emotion_statistics.append({
            'emotion': emotion,
            'count': count,
            'percentage': round(percentage, 2),
            'avg_confidence': round(float(avg_conf), 4)
        })
    
    # Sort by count descending
    emotion_statistics.sort(key=lambda x: x['count'], reverse=True)
    
    # Find dominant emotion
    dominant_emotion = None
    dominant_confidence = 0.0
    if emotion_statistics and emotion_statistics[0]['count'] > 0:
        dominant_emotion = emotion_statistics[0]['emotion']
        dominant_confidence = emotion_statistics[0]['avg_confidence']
    
    # Calculate overall confidence (weighted average)
    total_confidence = sum(
        count * np.mean(emotion_confidences[emotion])
        for emotion, count in emotion_counts.items()
        if emotion in emotion_confidences
    )
    overall_confidence = total_confidence / total_detected_frames if total_detected_frames > 0 else 0
    
    # Emotion timeline analysis (detect emotion changes)
    emotion_segments = []
    if frame_emotions:
        current_emotion = None
        segment_start = 0
        
        for i, frame_data in enumerate(frame_emotions):
            if frame_data['emotion'] != current_emotion and frame_data['emotion'] != 'No Face Detected':
                if current_emotion is not None:
                    emotion_segments.append({
                        'emotion': current_emotion,
                        'start_time': round(frame_emotions[segment_start]['timestamp'], 2),
                        'end_time': round(frame_emotions[i-1]['timestamp'], 2),
                        'duration': round(frame_emotions[i-1]['timestamp'] - frame_emotions[segment_start]['timestamp'], 2)
                    })
                current_emotion = frame_data['emotion']
                segment_start = i
        
        # Add final segment
        if current_emotion is not None and current_emotion != 'No Face Detected':
            emotion_segments.append({
                'emotion': current_emotion,
                'start_time': round(frame_emotions[segment_start]['timestamp'], 2),
                'end_time': round(frame_emotions[-1]['timestamp'], 2),
                'duration': round(frame_emotions[-1]['timestamp'] - frame_emotions[segment_start]['timestamp'], 2)
            })
    
    return {
        'video_metadata': {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': round(fps, 2),
            'duration_seconds': round(duration, 2),
            'resolution': f"{width}x{height}",
            'sample_rate': sample_rate
        },
        'detection_summary': {
            'frames_with_faces': frames_with_faces,
            'frames_without_faces': frames_without_faces,
            'face_detection_rate': round(frames_with_faces / processed_frames * 100, 2) if processed_frames > 0 else 0
        },
        'emotion_analysis': {
            'dominant_emotion': dominant_emotion,
            'dominant_confidence': round(float(dominant_confidence), 4),
            'overall_confidence': round(float(overall_confidence), 4),
            'total_detections': total_detected_frames,
            'emotion_statistics': emotion_statistics,
            'emotion_segments': emotion_segments[:20]  # Limit to first 20 segments
        },
        'frame_by_frame': frame_emotions[:100]  # Limit to first 100 frames for response size
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MMIA Video Emotion Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze video for emotions",
            "/analyze-with-transcription": "POST - Analyze video with audio transcription",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "face_detector": face_detector is not None,
            "emotion_predictor": emotion_predictor is not None,
            "whisper": whisper_model is not None
        }
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    sample_rate: int = 1,
    include_frames: bool = False
):
    """
    Analyze emotions in uploaded video
    
    Args:
        file: MP4 video file
        sample_rate: Process every Nth frame (default: 1)
        include_frames: Include frame-by-frame data in response (default: False)
        
    Returns:
        Comprehensive emotion analysis
    """
    start_time = datetime.now()
    
    logger.info(f"Received video analysis request: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload MP4, AVI, MOV, or MKV video."
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Analyze video
        results = analyze_video_emotions(tmp_path, sample_rate=sample_rate)
        
        # Remove frame-by-frame data if not requested
        if not include_frames:
            results.pop('frame_by_frame', None)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Analysis complete in {processing_time:.2f}s")
        
        return {
            "success": True,
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2),
            "analysis": results
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/analyze-with-transcription")
async def analyze_video_with_transcription(
    file: UploadFile = File(...),
    sample_rate: int = 1,
    include_frames: bool = False,
    transcribe: bool = True
):
    """
    Analyze emotions in uploaded video and transcribe audio
    
    Args:
        file: MP4 video file
        sample_rate: Process every Nth frame (default: 1)
        include_frames: Include frame-by-frame data in response (default: False)
        transcribe: Enable audio transcription (default: True)
        
    Returns:
        Comprehensive emotion analysis with transcription
    """
    start_time = datetime.now()
    
    logger.info(f"Received video + transcription request: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload MP4, AVI, MOV, or MKV video."
        )
    
    # Save uploaded file temporarily in the same directory as api_services.py
    video_tmp_path = os.path.join(SCRIPT_DIR, f"_tmp_video_{os.getpid()}.mp4")
    with open(video_tmp_path, 'wb') as tmp_video:
        content = await file.read()
        tmp_video.write(content)
    video_path = video_tmp_path
    
    audio_path = None
    transcription_result = None
    
    try:
        # Analyze video emotions
        logger.info("Starting emotion analysis...")
        emotion_results = analyze_video_emotions(video_path, sample_rate=sample_rate)
        
        # Extract and transcribe audio if requested
        if transcribe:
            audio_path = os.path.join(SCRIPT_DIR, f"_tmp_audio_{os.getpid()}.wav")
            
            logger.info("Extracting audio from video...")
            if extract_audio_from_video(video_path, audio_path):
                logger.info("Starting audio transcription...")
                transcription_result = transcribe_audio(audio_path)
            else:
                logger.warning("Could not extract audio from video")
                transcription_result = {
                    'success': False,
                    'error': 'No audio stream found in video or extraction failed',
                    'full_text': '',
                    'segments': [],
                    'segment_count': 0
                }
        
        # Remove frame-by-frame data if not requested
        if not include_frames:
            emotion_results.pop('frame_by_frame', None)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Complete analysis finished in {processing_time:.2f}s")
        
        # Combine results
        response = {
            "success": True,
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2),
            "emotion_analysis": emotion_results,
            "transcription": transcription_result if transcribe else None
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
    finally:
        # Clean up temporary files
        if os.path.exists(video_path):
            os.unlink(video_path)
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


@app.post("/batch-analyze")
async def batch_analyze_videos(
    files: List[UploadFile] = File(...),
    sample_rate: int = 2
):
    """
    Analyze multiple videos in batch
    
    Args:
        files: List of video files
        sample_rate: Process every Nth frame (default: 2)
        
    Returns:
        List of analysis results
    """
    logger.info(f"Batch analysis request: {len(files)} videos")
    
    results = []
    
    for file in files:
        try:
            # Process each video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            analysis = analyze_video_emotions(tmp_path, sample_rate=sample_rate)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "analysis": analysis
            })
            
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_videos": len(files),
        "successful": sum(1 for r in results if r['success'])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)