"""
NLP Analyzer for MMIA — Verbal Communication Analysis
======================================================
Loads models from local /models/nlp/ directory (no internet needed after first download).

Models used:
    - models/nlp/sentiment/   → cardiffnlp/twitter-roberta-base-sentiment-latest
    - models/nlp/zero_shot/   → facebook/bart-large-mnli

Paper alignment:
    Section 3.2 — Verbal Cue Analysis:
        "Sentiment analysis detects affective tone — positive, negative, or neutral —
         to infer candidate enthusiasm or hesitation."
        "Linguistic style matching examines syntactic and lexical patterns that reveal
         personality traits or communication styles."

Run download_nlp_models.py ONCE before using this module.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# IMPORTANT: Configure HuggingFace to use local models directory (D: drive)
# This prevents downloads to C:\Users\...\cache\huggingface\
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_NLP_DIR = _PROJECT_ROOT / "models" / "nlp"

os.environ["HF_HOME"] = str(_NLP_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(_NLP_DIR)
os.environ["HF_DATASETS_CACHE"] = str(_NLP_DIR / "datasets")

logger = logging.getLogger(__name__)

SENTIMENT_MODEL_PATH = str(_NLP_DIR / "sentiment")
ZERO_SHOT_MODEL_PATH = str(_NLP_DIR / "zero_shot")

# ── Lazy-loaded singletons (loaded once, reused across requests) ──────────────
_sentiment_pipeline  = None
_zero_shot_pipeline  = None

# ── Interview-specific classification labels ─────────────────────────────────
# These map directly to paper Section 3.2 "linguistic style" indicators.
INTERVIEW_STYLE_LABELS = [
    "confident",
    "hesitant",
    "enthusiastic",
    "nervous",
    "knowledgeable",
    "vague",
    "assertive",
    "uncertain",
]

# Filler words that signal hesitation / lack of preparation (paper Section 3.2)
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "er", "err",
    "like", "basically", "literally", "actually",
    "you know", "i mean", "kind of", "sort of",
    "right", "okay so", "so yeah",
}


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_sentiment_pipeline():
    """Load sentiment model from local disk. Falls back to HuggingFace Hub if not found."""
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    from transformers import pipeline as hf_pipeline

    local_path = Path(SENTIMENT_MODEL_PATH)
    if local_path.exists() and (local_path / "config.json").exists():
        source = SENTIMENT_MODEL_PATH
        logger.info(f"Loading sentiment model from local disk: {source}")
    else:
        source = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        logger.warning(
            f"Local sentiment model not found at {SENTIMENT_MODEL_PATH}. "
            f"Falling back to HuggingFace Hub (requires internet). "
            f"Run download_nlp_models.py to cache locally."
        )

    _sentiment_pipeline = hf_pipeline(
        "sentiment-analysis",
        model=source,
        tokenizer=source,
        top_k=None,           # return all 3 label scores, not just top-1
        truncation=True,
        max_length=512,
        device=-1,             # CPU inference
        model_kwargs={"cache_dir": str(_NLP_DIR)} if source.startswith("cardiffnlp") else {}
    )
    logger.info("✓ Sentiment model loaded")
    return _sentiment_pipeline


def _load_zero_shot_pipeline():
    """Load zero-shot classifier from local disk. Falls back to HuggingFace Hub if not found."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is not None:
        return _zero_shot_pipeline

    from transformers import pipeline as hf_pipeline

    local_path = Path(ZERO_SHOT_MODEL_PATH)
    if local_path.exists() and (local_path / "config.json").exists():
        source = ZERO_SHOT_MODEL_PATH
        logger.info(f"Loading zero-shot model from local disk: {source}")
    else:
        source = "facebook/bart-large-mnli"
        logger.warning(
            f"Local zero-shot model not found at {ZERO_SHOT_MODEL_PATH}. "
            f"Falling back to HuggingFace Hub (requires internet). "
            f"Run download_nlp_models.py to cache locally."
        )

    _zero_shot_pipeline = hf_pipeline(
        "zero-shot-classification",
        model=source,
        tokenizer=source,
        multi_label=True,
    )
    logger.info("✓ Zero-shot classifier loaded")
    return _zero_shot_pipeline


# ── Helper functions ──────────────────────────────────────────────────────────

def _count_filler_words(text: str) -> dict:
    """
    Count filler words in transcript.
    Returns count and list of fillers found.
    """
    text_lower = text.lower()
    words      = text_lower.split()
    
    found      = []
    count      = 0
    
    # Single-word fillers
    for word in words:
        clean = word.strip(".,!?;:")
        if clean in FILLER_WORDS:
            found.append(clean)
            count += 1
    
    # Multi-word fillers (phrases)
    for phrase in ["you know", "i mean", "kind of", "sort of", "okay so", "so yeah"]:
        occurrences = text_lower.count(phrase)
        if occurrences > 0:
            count += occurrences
            found.extend([phrase] * occurrences)
    
    return {
        "count":    count,
        "found":    list(set(found)),   # unique fillers detected
        "raw_list": found               # all occurrences
    }


def _normalize_label(label: str) -> str:
    """Normalize sentiment label to uppercase standard."""
    mapping = {
        "positive":  "POSITIVE",
        "negative":  "NEGATIVE",
        "neutral":   "NEUTRAL",
        "label_0":   "NEGATIVE",   # some model variants use label_0/1/2
        "label_1":   "NEUTRAL",
        "label_2":   "POSITIVE",
    }
    return mapping.get(label.lower(), label.upper())


# ── Main public function ──────────────────────────────────────────────────────

def analyze_transcript(text: str, top_n_styles: int = 4) -> dict:
    """
    Full NLP analysis of interview transcript.

    Args:
        text:          Full transcript string from Whisper
        top_n_styles:  How many top communication style labels to include

    Returns:
        dict with sentiment, communication style, filler analysis,
        word metrics, and recruiter-facing interpretations.

    Example output:
        {
          "sentiment": {
              "label": "POSITIVE",
              "score": 0.934,
              "all_scores": {
                  "POSITIVE": 0.934,
                  "NEUTRAL":  0.049,
                  "NEGATIVE": 0.017
              }
          },
          "communication_style": {
              "assertive":     0.891,
              "confident":     0.876,
              "knowledgeable": 0.743,
              "enthusiastic":  0.621
          },
          "filler_words": {
              "count":  2,
              "ratio":  0.018,
              "found":  ["um", "like"],
              "raw_list": ["um", "like"]
          },
          "word_metrics": {
              "word_count":         112,
              "unique_word_count":  78,
              "vocabulary_richness": 0.696,
              "avg_word_length":    5.2
          },
          "interpretation": {
              "clarity":        "high",
              "verbal_quality": "strong",
              "summary":        "Candidate communicates assertively and confidently..."
          },
          "verbal_score": 0.87      ← used by candidate_scorer.py
        }
    """
    # ── Input guard ───────────────────────────────────────────────────────────
    if not text or len(text.strip()) < 5:
        return {
            "error":        "Transcript too short for analysis",
            "verbal_score": 0.5
        }

    text = text.strip()

    # ── 1. Sentiment analysis ─────────────────────────────────────────────────
    try:
        sentiment_pipe   = _load_sentiment_pipeline()
        # top_k=None returns all labels with scores
        raw_sentiment    = sentiment_pipe(text[:512])[0]   # list of {label, score}

        # Build normalized score dict
        all_scores = {
            _normalize_label(item["label"]): round(item["score"], 4)
            for item in raw_sentiment
        }

        # Top label
        top_item      = max(raw_sentiment, key=lambda x: x["score"])
        top_label     = _normalize_label(top_item["label"])
        top_score     = round(top_item["score"], 4)

        sentiment_result = {
            "label":      top_label,
            "score":      top_score,
            "all_scores": all_scores,
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        sentiment_result = {
            "label":      "NEUTRAL",
            "score":      0.5,
            "all_scores": {"POSITIVE": 0.33, "NEUTRAL": 0.34, "NEGATIVE": 0.33},
            "error":      str(e)
        }

    # ── 2. Zero-shot communication style classification ───────────────────────
    try:
        zero_shot_pipe  = _load_zero_shot_pipeline()
        style_result    = zero_shot_pipe(
            text[:512],
            candidate_labels=INTERVIEW_STYLE_LABELS,
            multi_label=True
        )

        # Zip labels + scores, take top N
        style_scores = dict(zip(style_result["labels"], style_result["scores"]))
        top_styles   = dict(
            sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_styles]
        )
        top_styles   = {k: round(v, 4) for k, v in top_styles.items()}

    except Exception as e:
        logger.error(f"Zero-shot classification failed: {e}")
        top_styles = {"confident": 0.5, "error": str(e)}

    # ── 3. Filler word analysis ───────────────────────────────────────────────
    words         = text.split()
    word_count    = len(words)
    filler_data   = _count_filler_words(text)
    filler_ratio  = round(filler_data["count"] / max(word_count, 1), 4)

    filler_result = {
        "count":    filler_data["count"],
        "ratio":    filler_ratio,
        "found":    filler_data["found"],
        "raw_list": filler_data["raw_list"],
    }

    # ── 4. Word-level metrics ─────────────────────────────────────────────────
    unique_words      = set(w.lower().strip(".,!?;:") for w in words)
    vocab_richness    = round(len(unique_words) / max(word_count, 1), 4)
    avg_word_length   = round(
        sum(len(w.strip(".,!?;:")) for w in words) / max(word_count, 1), 2
    )

    word_metrics = {
        "word_count":         word_count,
        "unique_word_count":  len(unique_words),
        "vocabulary_richness": vocab_richness,   # higher = more diverse vocab
        "avg_word_length":    avg_word_length,   # proxy for lexical complexity
    }

    # ── 5. Interpretations ───────────────────────────────────────────────────
    # Clarity: based on filler ratio
    if filler_ratio < 0.03:
        clarity = "high"
    elif filler_ratio < 0.07:
        clarity = "moderate"
    else:
        clarity = "low"

    # Verbal quality: combined sentiment + style signal
    top_style_label = list(top_styles.keys())[0] if top_styles else "unknown"
    positive_styles = {"confident", "assertive", "enthusiastic", "knowledgeable"}
    negative_styles = {"hesitant", "nervous", "vague", "uncertain"}

    style_is_positive = top_style_label in positive_styles
    sentiment_is_positive = sentiment_result["label"] == "POSITIVE"

    if style_is_positive and sentiment_is_positive:
        verbal_quality = "strong"
    elif style_is_positive or sentiment_is_positive:
        verbal_quality = "good"
    elif top_style_label in negative_styles:
        verbal_quality = "needs improvement"
    else:
        verbal_quality = "average"

    # Auto-summary for recruiter dashboard
    summary_parts = []
    summary_parts.append(
        f"Candidate communicates in a {top_style_label} manner"
    )
    if sentiment_result["label"] == "POSITIVE":
        summary_parts.append("with a positive and enthusiastic tone")
    elif sentiment_result["label"] == "NEGATIVE":
        summary_parts.append("with a notably uncertain or negative tone")

    if filler_ratio < 0.03:
        summary_parts.append("and speaks clearly with minimal hesitation")
    elif filler_ratio > 0.07:
        summary_parts.append(f"but uses frequent filler words ({filler_data['count']} detected)")

    interpretation = {
        "clarity":        clarity,
        "verbal_quality": verbal_quality,
        "summary":        ". ".join(summary_parts) + ".",
    }

    # ── 6. Verbal score for candidate_scorer.py ──────────────────────────────
    # Formula (matches candidate_scorer.py):
    #   sentiment_component = 1.0 if POSITIVE | 0.6 if NEUTRAL | 0.2 if NEGATIVE
    #   filler_component    = max(0, 1 - filler_ratio × 5)
    #   verbal_score        = sentiment_component × 0.5 + filler_component × 0.5
    sentiment_component = (
        1.0 if sentiment_result["label"] == "POSITIVE" else
        0.6 if sentiment_result["label"] == "NEUTRAL"  else 0.2
    )
    filler_component    = max(0.0, 1.0 - filler_ratio * 5)
    verbal_score        = round(sentiment_component * 0.5 + filler_component * 0.5, 4)

    # ── Final response ────────────────────────────────────────────────────────
    return {
        "sentiment":           sentiment_result,
        "communication_style": top_styles,
        "filler_words":        filler_result,
        "word_metrics":        word_metrics,
        "interpretation":      interpretation,
        "verbal_score":        verbal_score,
    }