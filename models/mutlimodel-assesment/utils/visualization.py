"""
Visualization utilities for emotion recognition
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config.config import EMOTION_COLORS


def create_emotion_bar_chart(probabilities):
    """
    Create a bar chart for emotion probabilities
    
    Args:
        probabilities: Dict of emotion -> probability
        
    Returns:
        Plotly figure
    """
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create color list
    colors = [
        f'rgb{EMOTION_COLORS[emotion]}' 
        for emotion in emotions
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=colors,
            text=[f'{p:.2%}' for p in probs],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Current Emotion Probabilities",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_emotion_pie_chart(emotion_counts):
    """
    Create pie chart for overall emotion distribution
    
    Args:
        emotion_counts: Dict of emotion -> count
        
    Returns:
        Plotly figure
    """
    if not emotion_counts or sum(emotion_counts.values()) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Overall Emotion Distribution",
            annotations=[{
                'text': 'No data yet',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        return fig
    
    # Sort emotions by count
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    emotions = [e[0] for e in sorted_emotions]
    counts = [e[1] for e in sorted_emotions]
    
    colors = [
        f'rgb{EMOTION_COLORS[emotion]}' 
        for emotion in emotions
    ]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=emotions,
            values=counts,
            marker=dict(colors=colors),
            hole=0.3,
            textinfo='label+percent',
        )
    ])
    
    fig.update_layout(
        title="Overall Emotion Distribution",
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_emotion_timeline(emotion_history):
    """
    Create timeline of detected emotions
    
    Args:
        emotion_history: List of (timestamp, emotion, confidence) tuples
        
    Returns:
        Plotly figure
    """
    if not emotion_history:
        fig = go.Figure()
        fig.update_layout(
            title="Emotion Timeline",
            annotations=[{
                'text': 'No data yet',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        return fig
    
    df = pd.DataFrame(
        emotion_history,
        columns=['timestamp', 'emotion', 'confidence']
    )
    
    # Define all emotion labels
    all_emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
    
    fig = px.scatter(
        df,
        x='timestamp',
        y='emotion',
        size='confidence',
        color='emotion',
        color_discrete_map={
            emotion: f'rgb{EMOTION_COLORS[emotion]}'
            for emotion in all_emotions if emotion in EMOTION_COLORS
        },
        size_max=20,
        title="Emotion Detection Timeline"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Detected Emotion",
        template="plotly_white"
    )
    
    return fig


def format_statistics(emotion_counts, total_frames):
    """
    Format emotion statistics as a dataframe
    
    Args:
        emotion_counts: Dict of emotion -> count
        total_frames: Total number of frames processed
        
    Returns:
        Pandas DataFrame
    """
    if total_frames == 0:
        return pd.DataFrame(columns=['Emotion', 'Count', 'Percentage'])
    
    # Get all emotion labels
    all_emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
    
    data = []
    for emotion in all_emotions:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        data.append({
            'Emotion': emotion,
            'Count': count,
            'Percentage': f'{percentage:.1f}%'
        })
    
    df = pd.DataFrame(data)
    # Sort by count descending
    df = df.sort_values('Count', ascending=False).reset_index(drop=True)
    return df