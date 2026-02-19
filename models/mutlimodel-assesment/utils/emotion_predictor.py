"""
Emotion prediction using EMO-AffectNet model (original implementation)
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import deque

from utils.model_architectures import ResNet50, LSTMPyTorch
from config.config import (
    MODEL_BACKBONE_PATH,
    MODEL_LSTM_PATH,
    EMOTION_LABELS,
    IMG_SIZE,
    MEAN_VALUES,
    LSTM_SEQUENCE_LENGTH
)


class PreprocessInput(torch.nn.Module):
    """Custom preprocessing as in original implementation"""
    
    def __init__(self):
        super(PreprocessInput, self).__init__()

    def forward(self, x):
        """
        Preprocess input with BGR channel flip and mean subtraction
        x: tensor in RGB format
        """
        x = x.to(torch.float32)
        # Flip RGB to BGR
        x = torch.flip(x, dims=(0,))
        # Subtract mean values (BGR)
        x[0, :, :] -= MEAN_VALUES[0]  # B
        x[1, :, :] -= MEAN_VALUES[1]  # G
        x[2, :, :] -= MEAN_VALUES[2]  # R
        return x


class EmotionPredictor:
    """Emotion predictor with LSTM temporal smoothing"""
    
    def __init__(self, backbone_path=MODEL_BACKBONE_PATH, lstm_path=MODEL_LSTM_PATH, device=None):
        """
        Initialize emotion predictor
        
        Args:
            backbone_path: Path to ResNet50 backbone weights
            lstm_path: Path to LSTM model weights
            device: torch device (cuda/cpu)
        """
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load models
        self.backbone_model = self._load_backbone(backbone_path)
        self.lstm_model = self._load_lstm(lstm_path)
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            PreprocessInput()
        ])
        
        self.emotion_labels = EMOTION_LABELS
        
        # LSTM feature buffer
        self.lstm_features = deque(maxlen=LSTM_SEQUENCE_LENGTH)
        
    def _load_backbone(self, model_path):
        """Load ResNet50 backbone model"""
        model = ResNet50(num_classes=7, channels=3)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✓ Backbone model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load backbone model: {e}")
            print("Using randomly initialized backbone for demonstration")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_lstm(self, model_path):
        """Load LSTM model"""
        model = LSTMPyTorch()
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✓ LSTM model loaded from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load LSTM model: {e}")
            print("Using randomly initialized LSTM for demonstration")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_face(self, face_rgb):
        """
        Preprocess face image (expects RGB numpy array)
        
        Args:
            face_rgb: RGB face image (numpy array)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Resize to 224x224 with NEAREST interpolation (as in original)
        pil_image = pil_image.resize(IMG_SIZE, Image.Resampling.NEAREST)
        
        # Apply transforms (convert to tensor and preprocess)
        face_tensor = self.transform(pil_image)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_rgb):
        """
        Predict emotion from face image with LSTM temporal smoothing
        
        Args:
            face_rgb: RGB face image (numpy array)
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities_dict)
        """
        with torch.no_grad():
            # Preprocess face
            face_tensor = self.preprocess_face(face_rgb)
            
            # Extract features using backbone
            features = torch.nn.functional.relu(
                self.backbone_model.extract_features(face_tensor)
            ).cpu().numpy()
            
            # Initialize LSTM buffer if empty
            if len(self.lstm_features) == 0:
                # Fill with same features for first frame
                for _ in range(LSTM_SEQUENCE_LENGTH):
                    self.lstm_features.append(features)
            else:
                # Add new features to buffer
                self.lstm_features.append(features)
            
            # Prepare LSTM input (stack features)
            lstm_input = np.vstack(list(self.lstm_features))
            lstm_tensor = torch.from_numpy(lstm_input).unsqueeze(0).to(self.device)
            
            # Predict using LSTM
            output = self.lstm_model(lstm_tensor).cpu().numpy()[0]
            
            # Get prediction
            predicted_idx = np.argmax(output)
            emotion = self.emotion_labels[predicted_idx]
            confidence = output[predicted_idx]
            
            # Create probability dictionary
            prob_dict = {
                self.emotion_labels[i]: float(output[i])
                for i in range(len(output))
            }
            
            return emotion, confidence, prob_dict
    
    def reset_lstm_buffer(self):
        """Reset LSTM feature buffer"""
        self.lstm_features.clear()
    
    def predict_batch(self, face_images):
        """
        Predict emotions for multiple faces
        
        Args:
            face_images: List of RGB face images
            
        Returns:
            List of (emotion, confidence, probabilities) tuples
        """
        results = []
        for face_rgb in face_images:
            try:
                result = self.predict_emotion(face_rgb)
                results.append(result)
            except Exception as e:
                print(f"Error predicting emotion: {e}")
                # Return neutral as default
                default_probs = {label: 0.0 for label in self.emotion_labels.values()}
                default_probs["Neutral"] = 1.0
                results.append(("Neutral", 0.0, default_probs))
        
        return results