WEIGHTS = {
    "emotion": 0.30,
    "eye_contact": 0.20,
    "acoustic": 0.25,
    "verbal": 0.25,
}

POSITIVE_EMOTIONS = {"Happiness", "Surprise"}
NEGATIVE_EMOTIONS = {"Fear", "Disgust", "Anger", "Sadness"}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _grade_from_score(composite):
    if composite > 0.8:
        return "Excellent"
    if composite > 0.6:
        return "Good"
    if composite > 0.4:
        return "Average"
    return "Needs Review"


def _is_emotion_available(emotion_analysis):
    if not emotion_analysis:
        return False
    total = _safe_float(emotion_analysis.get("total_detections"), 0.0)
    stats = emotion_analysis.get("emotion_statistics") or []
    return total > 0 and len(stats) > 0


def _is_gaze_available(gaze_analysis):
    return bool(gaze_analysis and gaze_analysis.get("gaze_frames_analyzed", 0) > 0)


def _is_acoustic_available(acoustic_analysis):
    if not acoustic_analysis or acoustic_analysis.get("error"):
        return False
    if acoustic_analysis.get("available") is False:
        return False
    if acoustic_analysis.get("speech_detected") is False:
        return False
    return True


def _is_verbal_available(nlp_analysis):
    if not nlp_analysis or nlp_analysis.get("error"):
        return False
    if nlp_analysis.get("available") is False:
        return False
    return True


def compute_candidate_score(emotion_analysis, gaze_analysis, acoustic_analysis, nlp_analysis):
    scores = {}
    availability = {}

    # 1) Emotion score
    emotion_stats = {}
    if _is_emotion_available(emotion_analysis):
        emotion_stats = {
            entry.get("emotion"): _safe_float(entry.get("percentage"), 0.0)
            for entry in (emotion_analysis.get("emotion_statistics") or [])
            if entry.get("emotion")
        }
        positive_pct = sum(emotion_stats.get(e, 0.0) for e in POSITIVE_EMOTIONS)
        negative_pct = sum(emotion_stats.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
        scores["emotion"] = min(1.0, max(0.0, (positive_pct / 100.0) * 0.6 + (1.0 - negative_pct / 100.0) * 0.4))
        availability["emotion"] = True
    else:
        scores["emotion"] = None
        availability["emotion"] = False

    # 2) Eye contact score
    if _is_gaze_available(gaze_analysis):
        eye_contact_pct = _safe_float(gaze_analysis.get("eye_contact_percentage"), 0.0)
        scores["eye_contact"] = min(1.0, max(0.0, eye_contact_pct / 100.0))
        availability["eye_contact"] = True
    else:
        scores["eye_contact"] = None
        availability["eye_contact"] = False

    # 3) Acoustic score (align with actual nested keys produced by acoustic_analyzer.py)
    if _is_acoustic_available(acoustic_analysis):
        pitch_std = _safe_float((acoustic_analysis.get("pitch") or {}).get("std_hz"), 0.0)
        energy = _safe_float((acoustic_analysis.get("energy") or {}).get("mean"), 0.0)
        pitch_score = min(1.0, max(0.0, pitch_std / 30.0))
        energy_score = min(1.0, max(0.0, energy / 0.08))
        scores["acoustic"] = (pitch_score + energy_score) / 2.0
        availability["acoustic"] = True
    else:
        scores["acoustic"] = None
        availability["acoustic"] = False

    # 4) Verbal/NLP score
    if _is_verbal_available(nlp_analysis):
        sentiment_label = ((nlp_analysis.get("sentiment") or {}).get("label") or "NEUTRAL").upper()
        filler_ratio = _safe_float((nlp_analysis.get("filler_words") or {}).get("ratio"), 0.0)
        sentiment_component = 1.0 if sentiment_label == "POSITIVE" else 0.6 if sentiment_label == "NEUTRAL" else 0.2
        filler_component = max(0.0, 1.0 - filler_ratio * 5.0)
        scores["verbal"] = (sentiment_component * 0.5) + (filler_component * 0.5)
        availability["verbal"] = True
    else:
        scores["verbal"] = None
        availability["verbal"] = False

    # Re-normalize weights to available modalities only
    available_dims = [name for name, ok in availability.items() if ok]
    if available_dims:
        total_weight = sum(WEIGHTS[name] for name in available_dims)
        effective_weights = {
            name: round((WEIGHTS[name] / total_weight), 4) if name in available_dims else 0.0
            for name in WEIGHTS
        }
        composite = sum(effective_weights[name] * scores[name] for name in available_dims)
    else:
        effective_weights = {name: 0.0 for name in WEIGHTS}
        composite = 0.0

    dimension_scores = {
        name: round(score * 100.0, 1) if score is not None else None
        for name, score in scores.items()
    }

    return {
        "composite_score": round(composite * 100.0, 1),
        "grade": _grade_from_score(composite),
        "dimension_scores": dimension_scores,
        "weights_used": effective_weights,
        "base_weights": WEIGHTS,
        "availability": availability,
        "highlights": _generate_highlights(emotion_stats, gaze_analysis, acoustic_analysis, nlp_analysis),
    }


def _generate_highlights(emotion_stats, gaze, acoustic, nlp):
    out = []
    if emotion_stats.get("Happiness", 0) > 30:
        out.append("Candidate displayed strong positive affect throughout")
    if gaze and gaze.get("eye_contact_percentage", 0) > 70:
        out.append("Maintained consistent eye contact, indicating engagement")
    if nlp and not nlp.get("error"):
        if (nlp.get("filler_words") or {}).get("ratio", 1) < 0.03:
            out.append("Clear, articulate speech with minimal hesitation")
    if acoustic and not acoustic.get("error"):
        if ((acoustic.get("pitch") or {}).get("std_hz", 0) > 20):
            out.append("Varied vocal tone indicating enthusiasm")
    return out
