# New util: utils/candidate_scorer.py

WEIGHTS = {
    "emotion":     0.30,   # facial FER (your two models)
    "eye_contact": 0.20,   # gaze consistency
    "acoustic":    0.25,   # vocal confidence
    "verbal":      0.25    # NLP quality
}

POSITIVE_EMOTIONS = {"Happiness", "Surprise"}
NEGATIVE_EMOTIONS = {"Fear", "Disgust", "Anger", "Sadness"}

def compute_candidate_score(emotion_analysis, gaze_analysis,
                             acoustic_analysis, nlp_analysis):
    scores = {}
    
    # 1. Emotion score — reward positive, penalise sustained negative
    dom = emotion_analysis.get("dominant_emotion", "Neutral")
    conf = emotion_analysis.get("overall_confidence", 0)
    emotion_stats = {e["emotion"]: e["percentage"]
                     for e in emotion_analysis.get("emotion_statistics", [])}
    
    positive_pct = sum(emotion_stats.get(e, 0) for e in POSITIVE_EMOTIONS)
    negative_pct = sum(emotion_stats.get(e, 0) for e in NEGATIVE_EMOTIONS)
    scores["emotion"] = min(1.0, max(0.0,
        (positive_pct / 100) * 0.6 + (1 - negative_pct / 100) * 0.4
    ))
    
    # 2. Eye contact score
    if gaze_analysis:
        eye_contact_pct = gaze_analysis.get("eye_contact_percentage", 50)
        scores["eye_contact"] = eye_contact_pct / 100
    else:
        scores["eye_contact"] = 0.5  # neutral if not available
    
    # 3. Acoustic score — pitch variation = engagement, energy = confidence
    if acoustic_analysis and not acoustic_analysis.get("error"):
        pitch_std  = acoustic_analysis.get("pitch_std_hz", 0)
        energy     = acoustic_analysis.get("energy_mean", 0)
        pitch_score  = min(1.0, pitch_std / 30)   # 30 Hz std = fully varied
        energy_score = min(1.0, energy / 0.08)
        scores["acoustic"] = (pitch_score + energy_score) / 2
    else:
        scores["acoustic"] = 0.5
    
    # 4. Verbal/NLP score
    if nlp_analysis and not nlp_analysis.get("error"):
        filler_penalty = nlp_analysis.get("filler_ratio", 0)
        sentiment_positive = 1.0 if nlp_analysis.get("sentiment") == "POSITIVE" else 0.4
        verbal_score = (sentiment_positive * 0.5) + (max(0, 1 - filler_penalty * 5) * 0.5)
        scores["verbal"] = verbal_score
    else:
        scores["verbal"] = 0.5
    
    # Weighted composite
    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    
    return {
        "composite_score":     round(composite * 100, 1),  # 0-100
        "grade": "Excellent" if composite > 0.8 else
                 "Good"      if composite > 0.6 else
                 "Average"   if composite > 0.4 else "Needs Review",
        "dimension_scores": {k: round(v * 100, 1) for k, v in scores.items()},
        "weights_used": WEIGHTS,
        "highlights": _generate_highlights(emotion_stats, gaze_analysis,
                                           acoustic_analysis, nlp_analysis)
    }

def _generate_highlights(emotion_stats, gaze, acoustic, nlp):
    out = []
    if emotion_stats.get("Happiness", 0) > 30:
        out.append("Candidate displayed strong positive affect throughout")
    if gaze and gaze.get("eye_contact_percentage", 0) > 70:
        out.append("Maintained consistent eye contact — signals engagement")
    if nlp and nlp.get("filler_ratio", 1) < 0.03:
        out.append("Clear, articulate speech with minimal hesitation")
    if acoustic and acoustic.get("pitch_std_hz", 0) > 20:
        out.append("Varied vocal tone indicating enthusiasm")
    return out