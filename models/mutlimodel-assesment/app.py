"""
Real-time Face Emotion Recognition using Streamlit
Based on EMO-AffectNet Model
"""
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
from collections import defaultdict, deque

from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor
from utils.visualization import (
    create_emotion_bar_chart,
    create_emotion_pie_chart,
    create_emotion_timeline,
    format_statistics
)
from config.config import EMOTION_LABELS


# Page configuration
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'emotion_counts' not in st.session_state:
        st.session_state.emotion_counts = defaultdict(int)
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = deque(maxlen=100)
    if 'face_detector' not in st.session_state:
        st.session_state.face_detector = FaceDetector()
    if 'emotion_predictor' not in st.session_state:
        with st.spinner("Loading emotion recognition model..."):
            st.session_state.emotion_predictor = EmotionPredictor()
    if 'running' not in st.session_state:
        st.session_state.running = False


def reset_statistics():
    """Reset all statistics"""
    st.session_state.emotion_counts = defaultdict(int)
    st.session_state.total_frames = 0
    st.session_state.emotion_history = deque(maxlen=100)


def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">😊 Real-time Face Emotion Recognition</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses the **EMO-AffectNet** model to detect emotions in real-time from your webcam.
    The model can recognize 7 different emotions: Neutral, Happy, Sad, Surprise, Fear, Disgust, and Anger.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Camera selection
        camera_index = st.selectbox(
            "Select Camera",
            options=[0, 1, 2],
            index=0,
            help="Select which camera to use"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence to display emotion"
        )
        
        # Show probabilities
        show_probabilities = st.checkbox(
            "Show All Probabilities",
            value=True,
            help="Display probability chart for all emotions"
        )
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Stats", use_container_width=True):
                reset_statistics()
                st.success("Statistics reset!")
        
        with col2:
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                **Model:** EMO-AffectNet  
                **Architecture:** ResNet50  
                **Dataset:** AffectNet  
                **Emotions:** 7 categories
                """)
        
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        st.metric("Total Frames", st.session_state.total_frames)
        
        if st.session_state.emotion_counts:
            most_common = max(
                st.session_state.emotion_counts.items(),
                key=lambda x: x[1]
            )
            st.metric("Most Common Emotion", most_common[0], most_common[1])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📈 Current Emotion")
        emotion_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if show_probabilities:
            st.subheader("📊 Probability Distribution")
            chart_placeholder = st.empty()
    
    # Statistics section
    st.markdown("---")
    st.subheader("📈 Overall Statistics")
    
    stats_col1, stats_col2 = st.columns([1, 1])
    
    with stats_col1:
        pie_chart_placeholder = st.empty()
    
    with stats_col2:
        stats_table_placeholder = st.empty()
    
    # Timeline
    timeline_placeholder = st.empty()
    
    # Start/Stop button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎥 Start/Stop Camera", use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running
    
    # Main video processing loop
    if st.session_state.running:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        status_placeholder.success("🟢 Camera is running...")
        
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                
                if not ret:
                    status_placeholder.error("❌ Failed to read from camera")
                    break
                
                # Detect faces
                faces = st.session_state.face_detector.detect_faces(frame)
                
                emotions_data = []
                current_emotion = None
                current_confidence = 0.0
                current_probabilities = {}
                
                if len(faces) > 0:
                    # Extract and predict emotions for each face
                    for bbox in faces:
                        face_img = st.session_state.face_detector.extract_face(
                            frame, bbox
                        )
                        
                        emotion, confidence, probabilities = \
                            st.session_state.emotion_predictor.predict_emotion(face_img)
                        
                        if confidence >= confidence_threshold:
                            emotions_data.append((emotion, confidence))
                            
                            # Update statistics for the first face
                            if current_emotion is None:
                                current_emotion = emotion
                                current_confidence = confidence
                                current_probabilities = probabilities
                                
                                # Update counts
                                st.session_state.emotion_counts[emotion] += 1
                                st.session_state.total_frames += 1
                                
                                # Add to history
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.session_state.emotion_history.append(
                                    (timestamp, emotion, confidence)
                                )
                    
                    # Draw bounding boxes
                    frame = st.session_state.face_detector.draw_face_boxes(
                        frame, faces, emotions_data
                    )
                
                # Display video frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update emotion display
                if current_emotion:
                    emotion_placeholder.markdown(
                        f"<h1 style='text-align: center; color: #1f77b4;'>{current_emotion}</h1>",
                        unsafe_allow_html=True
                    )
                    confidence_placeholder.markdown(
                        f"<h3 style='text-align: center;'>Confidence: {current_confidence:.2%}</h3>",
                        unsafe_allow_html=True
                    )
                    
                    # Update probability chart (throttled to reduce lag)
                    if show_probabilities and current_probabilities and st.session_state.total_frames % 3 == 0:
                        fig = create_emotion_bar_chart(current_probabilities)
                        chart_placeholder.plotly_chart(fig, width='stretch')
                else:
                    emotion_placeholder.markdown(
                        "<h3 style='text-align: center;'>No face detected</h3>",
                        unsafe_allow_html=True
                    )
                
                # Update statistics every 10 frames
                if st.session_state.total_frames % 10 == 0:
                    # Pie chart
                    pie_fig = create_emotion_pie_chart(st.session_state.emotion_counts)
                    pie_chart_placeholder.plotly_chart(pie_fig, width='stretch')
                    
                    # Statistics table
                    stats_df = format_statistics(
                        st.session_state.emotion_counts,
                        st.session_state.total_frames
                    )
                    stats_table_placeholder.dataframe(
                        stats_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Timeline
                    if len(st.session_state.emotion_history) > 0:
                        timeline_fig = create_emotion_timeline(
                            list(st.session_state.emotion_history)
                        )
                        timeline_placeholder.plotly_chart(
                            timeline_fig,
                            width='stretch'
                        )
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)
                
        finally:
            cap.release()
            status_placeholder.info("⭕ Camera stopped")
    else:
        status_placeholder.info("⭕ Camera is not running. Click 'Start/Stop Camera' to begin.")


if __name__ == "__main__":
    main()