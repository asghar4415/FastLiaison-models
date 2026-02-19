"""
Face detection utilities using MediaPipe (as in original implementation)
"""
import cv2
import numpy as np
import mediapipe as mp
import math
from config.config import (
    FACE_DETECTION_CONFIDENCE,
    FACE_TRACKING_CONFIDENCE,
    MAX_NUM_FACES
)


class FaceDetector:
    """Face detector using MediaPipe Face Mesh"""
    
    def __init__(self):
        """Initialize MediaPipe face mesh detector"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=False,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE
        )
    
    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        """Convert normalized coordinates to pixel coordinates"""
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    
    def get_box(self, face_landmarks, width, height):
        """Extract bounding box from face landmarks"""
        idx_to_coors = {}
        
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, width, height)
            if landmark_px:
                idx_to_coors[idx] = landmark_px

        # Get bounding box coordinates
        x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
        y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
        endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
        endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

        startX = max(0, x_min)
        startY = max(0, y_min)
        endX = min(width - 1, endX)
        endY = min(height - 1, endY)
        
        return startX, startY, endX, endY
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using MediaPipe
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of face bounding boxes [(startX, startY, endX, endY), ...]
            and the processed results
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process frame
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                box = self.get_box(face_landmarks, w, h)
                faces.append(box)
        
        return faces
    
    def extract_face(self, frame, bbox):
        """
        Extract face region from frame
        
        Args:
            frame: BGR image
            bbox: (startX, startY, endX, endY) bounding box
            
        Returns:
            Cropped face image in RGB format
        """
        startX, startY, endX, endY = bbox
        
        # Convert to RGB and crop
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = frame_rgb[startY:endY, startX:endX]
        
        return face
    
    def draw_face_boxes(self, frame, faces, emotions=None):
        """
        Draw bounding boxes and emotion labels on frame
        
        Args:
            frame: BGR image
            faces: List of face bounding boxes
            emotions: List of (emotion, confidence) tuples (optional)
            
        Returns:
            Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        for i, (startX, startY, endX, endY) in enumerate(faces):
            # Draw rectangle
            color = (255, 0, 255)  # Magenta as in original
            thickness = max(round(sum(frame.shape) / 2 * 0.003), 2)
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), color, thickness=thickness, lineType=cv2.LINE_AA)
            
            # Add emotion label if available
            if emotions and i < len(emotions):
                emotion, confidence = emotions[i]
                label = f"{emotion} {confidence:.1%}"
                
                # Calculate text position (centered above face)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = thickness / 3
                font_thickness = max(thickness - 1, 1)
                
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_width = text_size[0]
                center_face = startX + round((endX - startX) / 2)
                
                text_x = center_face - round(text_width / 2)
                text_y = startY - round(((endX - startX) * 20) / 360)
                
                # Draw text with black background
                cv2.putText(frame_copy, label, (text_x, text_y), font, font_scale, 
                           (0, 0, 0), thickness=font_thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame_copy, label, (text_x, text_y), font, font_scale, 
                           color, thickness=font_thickness, lineType=cv2.LINE_AA)
        
        return frame_copy
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()