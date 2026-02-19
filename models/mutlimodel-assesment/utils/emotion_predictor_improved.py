"""
Improved Emotion prediction with validated confidence calculations
Ensures proper softmax probabilities and confidence metrics
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


class ImprovedEmotionPredictor:
    """
    Enhanced emotion predictor with validated confidence calculations
    
    Key improvements:
    - Proper softmax probability validation
    - Confidence metrics validation
    - Better error handling
    - Detailed probability distribution
    """
    
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
        
        print(f"🔧 Using device: {self.device}")
        
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
        
        # Statistics tracking
        self.prediction_count = 0
        self.confidence_history = []
        
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
        # Validate input
        if face_rgb is None or face_rgb.size == 0:
            raise ValueError("Invalid face image: empty or None")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Resize to 224x224 with NEAREST interpolation (as in original)
        pil_image = pil_image.resize(IMG_SIZE, Image.Resampling.NEAREST)
        
        # Apply transforms (convert to tensor and preprocess)
        face_tensor = self.transform(pil_image)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def _validate_probabilities(self, probabilities):
        """
        Validate that probabilities sum to ~1.0 and are in valid range
        
        Args:
            probabilities: numpy array of probabilities
            
        Returns:
            Boolean indicating if probabilities are valid
        """
        # Check if all values are in [0, 1]
        if not np.all((probabilities >= 0) & (probabilities <= 1)):
            return False
        
        # Check if sum is close to 1.0 (within tolerance)
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            print(f"⚠️  Warning: Probabilities sum to {prob_sum:.6f}, not 1.0")
            return False
        
        return True
    
    def predict_emotion(self, face_rgb, return_detailed=False):
        """
        Predict emotion from face image with LSTM temporal smoothing
        
        Args:
            face_rgb: RGB face image (numpy array)
            return_detailed: Return detailed analysis including entropy, top-k, etc.
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities_dict)
            or detailed dict if return_detailed=True
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
            
            # Predict using LSTM (already applies softmax in forward pass)
            output = self.lstm_model(lstm_tensor).cpu().numpy()[0]
            
            # Validate probabilities
            is_valid = self._validate_probabilities(output)
            if not is_valid:
                print(f"⚠️  Invalid probability distribution detected")
                # Renormalize if needed
                output = output / np.sum(output)
            
            # Get prediction
            predicted_idx = np.argmax(output)
            emotion = self.emotion_labels[predicted_idx]
            confidence = float(output[predicted_idx])
            
            # Create probability dictionary (sorted by value)
            prob_dict = {
                self.emotion_labels[i]: float(output[i])
                for i in range(len(output))
            }
            
            # Update statistics
            self.prediction_count += 1
            self.confidence_history.append(confidence)
            
            if not return_detailed:
                return emotion, confidence, prob_dict
            
            # Calculate additional metrics for detailed analysis
            # 1. Entropy (measure of uncertainty)
            entropy = -np.sum(output * np.log(output + 1e-10))
            
            # 2. Top-3 emotions
            top_indices = np.argsort(output)[-3:][::-1]
            top_3 = [
                {
                    'emotion': self.emotion_labels[idx],
                    'probability': float(output[idx])
                }
                for idx in top_indices
            ]
            
            # 3. Confidence margin (difference between top 2)
            sorted_probs = np.sort(output)[::-1]
            confidence_margin = float(sorted_probs[0] - sorted_probs[1])
            
            # 4. Confidence category
            if confidence >= 0.7:
                confidence_level = "High"
            elif confidence >= 0.4:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'confidence_margin': confidence_margin,
                'all_probabilities': prob_dict,
                'top_3_emotions': top_3,
                'entropy': float(entropy),
                'is_valid_distribution': is_valid,
                'prediction_number': self.prediction_count
            }
    
    def reset_lstm_buffer(self):
        """Reset LSTM feature buffer and statistics"""
        self.lstm_features.clear()
        self.prediction_count = 0
        self.confidence_history = []
    
    def get_statistics(self):
        """Get prediction statistics"""
        if not self.confidence_history:
            return {
                'total_predictions': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'std_confidence': 0.0
            }
        
        return {
            'total_predictions': self.prediction_count,
            'avg_confidence': float(np.mean(self.confidence_history)),
            'min_confidence': float(np.min(self.confidence_history)),
            'max_confidence': float(np.max(self.confidence_history)),
            'std_confidence': float(np.std(self.confidence_history))
        }
    
    def predict_batch(self, face_images, return_detailed=False):
        """
        Predict emotions for multiple faces
        
        Args:
            face_images: List of RGB face images
            return_detailed: Return detailed analysis
            
        Returns:
            List of prediction results
        """
        results = []
        for face_rgb in face_images:
            try:
                result = self.predict_emotion(face_rgb, return_detailed=return_detailed)
                results.append(result)
            except Exception as e:
                print(f"Error predicting emotion: {e}")
                # Return neutral as default
                default_probs = {label: 0.0 for label in self.emotion_labels.values()}
                default_probs["Neutral"] = 1.0
                
                if return_detailed:
                    results.append({
                        'emotion': "Neutral",
                        'confidence': 0.0,
                        'confidence_level': "Error",
                        'all_probabilities': default_probs,
                        'error': str(e)
                    })
                else:
                    results.append(("Neutral", 0.0, default_probs))
        
        return results


# Maintain backward compatibility
EmotionPredictor = ImprovedEmotionPredictor