"""
Configuration file for emotion recognition application
"""
import os

# Base directory = the mutlimodel-assesment/ folder (parent of config/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths — absolute so they work from any CWD (gateway, direct run, etc.)
MODEL_BACKBONE_PATH = os.path.join(_BASE_DIR, "models", "FER_static_ResNet50_AffectNet.pt")
MODEL_LSTM_PATH     = os.path.join(_BASE_DIR, "models", "FER_dinamic_LSTM_Aff-Wild2.pt")

# Emotion Labels (AffectNet 7 emotions)
EMOTION_LABELS = {
    0: 'Neutral',
    1: 'Happiness', 
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger'
}

# Emotion Colors for visualization (RGB format)
EMOTION_COLORS = {
    "Neutral": (128, 128, 128),
    "Happiness": (255, 215, 0),
    "Sadness": (70, 130, 180),
    "Surprise": (255, 165, 0),
    "Fear": (138, 43, 226),
    "Disgust": (0, 128, 0),
    "Anger": (220, 20, 60)
}

# Model Input Configuration (from original notebook)
IMG_SIZE = (224, 224)
# BGR mean values from original preprocessing
MEAN_VALUES = [91.4953, 103.8827, 131.0912]  # BGR format

# MediaPipe Face Detection Configuration
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 1

# LSTM Configuration
LSTM_SEQUENCE_LENGTH = 10  # Number of frames for temporal smoothing

# Webcam Configuration
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
FPS = 30

# UI Configuration
CONFIDENCE_THRESHOLD = 0.3
UPDATE_INTERVAL = 0.1  # seconds