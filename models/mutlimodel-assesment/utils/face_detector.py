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
    
    def detect_faces_with_landmarks(self, frame):
        """
        Detect faces and return both bounding boxes and landmarks
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Tuple of (face_boxes, face_landmarks_list)
            face_boxes: List of bounding boxes [(startX, startY, endX, endY), ...]
            face_landmarks_list: List of MediaPipe face landmarks objects
        """
        import logging
        logger = logging.getLogger(__name__)
        
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process frame
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        faces = []
        landmarks_list = []
        if results.multi_face_landmarks:
            logger.debug(f"FaceMesh detected {len(results.multi_face_landmarks)} faces")
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                box = self.get_box(face_landmarks, w, h)
                faces.append(box)
                landmarks_list.append(face_landmarks)
                logger.debug(f"Face {i}: bbox={box}, landmarks available")
        else:
            logger.debug("No faces detected by FaceMesh")
        
        return faces, landmarks_list
    
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

    # In face_detector.py — add alongside existing detect_faces()

    def estimate_gaze_and_pose(self, frame, face_landmarks, w, h):
        """Estimate head pose and eye contact from face landmarks"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Key landmark indices for gaze estimation
            LEFT_EYE  = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            
            # Eye contact proxy: ratio of visible iris area
            # Head pose: solvePnP on 6 canonical 3D face points
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 33, 61, 199, 263, 291]:  # nose, eyes, chin
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            logger.debug(f"Landmarks extracted: {len(face_3d)} points")
            
            if len(face_3d) < 4:
                logger.warning("Insufficient landmarks for pose estimation")
                return None
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal = w  # approx focal length
            cam_matrix = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
            dist = np.zeros((4,1))
            
            success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
            
            if not success:
                logger.warning("solvePnP failed")
                return None
            
            rmat, _ = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(rmat)
            
            pitch, yaw, roll = angles  # degrees
            # yaw: left/right, pitch: up/down
            looking_at_camera = abs(yaw) < 10 and abs(pitch) < 10
            
            result = {
                "pitch": round(float(pitch), 2),
                "yaw": round(float(yaw), 2),
                "roll": round(float(roll), 2),
                "eye_contact": looking_at_camera
            }
            logger.debug(f"Gaze estimation result: {result}")
            return result
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in estimate_gaze_and_pose: {e}", exc_info=True)
            return None
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()