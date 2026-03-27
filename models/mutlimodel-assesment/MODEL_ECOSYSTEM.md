# MMIA Model Ecosystem & Feature Map

## Complete Model Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MMIA SYSTEM ARCHITECTURE                              │
│                     (All Components Implemented)                          │
└─────────────────────────────────────────────────────────────────────────┘

VIDEO INPUT
    │
    ├──────────────────────────────────────────────────────────────┐
    │                                                                 │
    ▼                                                                 ▼
┌──────────────────┐                              ┌─────────────────────┐
│ FACIAL ANALYSIS  │                              │ AUDIO EXTRACTION    │
│    (Video       │                              │   (moviepy)         │
│   Frames)       │                              │                     │
└──────────────────┘                              └─────────────────────┘
    │                                                    │
    ├─► MediaPipe                                        ├─► 16kHz PCM
    │   FaceMesh                                         │   mono WAV
    │   (468 landmarks)                                  │
    │                                                    ▼
    ├─► ResNet-50 + LSTM            ┌──────────────────────────────┐
    │   Training Data:               │  TRANSCRIPTION (Whisper)     │
    │   • AffectNet (450k faces)     │  Training Data:              │
    │   • Aff-Wild2 (video)          │  • 680k hours multilingual   │
    │                                │  • 99+ languages             │
    ├─► Head Pose (solvePnP)        └──────────────────────────────┘
    │   Pitch/Yaw/Roll                      │
    │                                        ▼
    └─► Eye Contact                 ┌──────────────────────────────┐
        (yaw/pitch thresholds)      │ ACOUSTIC ANALYSIS (librosa) │
                                     │ • Pitch (F0) extraction     │
    ▼                               │ • Energy (RMS)              │
┌──────────────────────────────┐   │ • Speech rate (ZCR)         │
│ EMOTION CLASSIFICATION       │   │ • Voice quality (MFCC)      │
│ Output: 7 emotions           │   │ • Voiced ratio              │
│ • Neutral                    │   │ Signal Processing (No NN)   │
│ • Happiness                  │   └──────────────────────────────┘
│ • Sadness                    │         │
│ • Surprise                   │         ▼
│ • Fear                       │   ┌──────────────────────────────┐
│ • Disgust                    │   │ NLP ANALYSIS                │
│ • Anger                      │   │                              │
│ Per-frame + Temporal         │   ├─► RoBERTa (Sentiment)      │
└──────────────────────────────┘   │   Training Data:            │
    │◄─────────────────────────────┤   • Twitter (680k tweets)   │
    │                              │   • Output: Pos/Neg/Neutral │
    │                              │                              │
    │                              ├─► BART-MNLI (Style)        │
    │                              │   Training Data:            │
    │                              │   • MNLI (433k pairs)       │
    │                              │   • 8-label zero-shot       │
    │                              │   • No fine-tuning needed   │
    │                              │                              │
    │                              ├─► Filler Detection         │
    │                              │   • Regex + word list       │
    │                              │   • Clarity metrics         │
    │                              └──────────────────────────────┘
    │                                   │
    └───────────────────┬───────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────┐
    │    COMPOSITE SCORE FUSION (WeightSum)    │
    │                                           │
    │  Final Score = Weighted Average:         │
    │  • Emotion:     30%  ← 7-class probs    │
    │  • Gaze/Eye:    20%  ← yaw/pitch        │
    │  • Acoustic:    25%  ← pitch var/energy│
    │  • Verbal:      25%  ← sentiment/NLP   │
    │                                           │
    │  Output: 0–100 score + Grade            │
    │  • Excellent  (80+)                     │
    │  • Good       (60–79)                   │
    │  • Average    (40–59)                   │
    │  • Needs Rev  (0–39)                    │
    │                                           │
    │  + Smart Highlights (3–4 per candidate) │
    └──────────────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────┐
    │  JSON RESPONSE (Complete Assessment)     │
    │  • Candidate metadata                    │
    │  • Composite score + grade               │
    │  • 4-dimensional breakdown               │
    │  • Recruiter highlights                  │
    │  • Processing time                       │
    └──────────────────────────────────────────┘
```

---

## Model Role Matrix

### By Capability

| Capability | Model | Size | Training Data | Type | Pre-trained? |
|-----------|-------|------|---------------|------|-------------|
| **Face Detection** | MediaPipe FaceMesh | 1 MB | Generic faces | Landmark | ✅ |
| **Emotion (Static)** | ResNet-50 | 103 MB | AffectNet (450k) | CNN | ✅ |
| **Emotion (Dynamic)** | LSTM | 45 MB | Aff-Wild2 | RNN | ✅ |
| **Gaze Estimation** | OpenCV solvePnP | — | Algorithm | Geometric | N/A |
| **Transcription** | Whisper Base | 140 MB | 680k hours | Transformer | ✅ |
| **Sentiment** | RoBERTa | 498 MB | Twitter (680k) | Transformer | ✅ |
| **Style Classify** | BART-large-MNLI | 1.63 GB | MNLI (433k) | Transformer | ✅ |
| **Acoustic** | librosa | — | Algorithm | Signal Proc | N/A |

---

### By Modality

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    MODALITY BREAKDOWN                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  VISUAL (Facial)                ACOUSTIC (Speech)                   ║
║  ───────────────                ──────────────────                  ║
║  ├─ MediaPipe (landmarks)      ├─ Whisper (transcription)         ║
║  ├─ ResNet-50 (emotion)        ├─ librosa (pitch/energy)          ║
║  ├─ LSTM (temporal)            ├─ RoBERTa (sentiment)             ║
║  └─ solvePnP (head pose)       ├─ BART (communication style)      ║
║                                 └─ Filler detection                 ║
║                                                                       ║
║  Weight: 30% + 20% (emotion+gaze) = 50%                            ║
║  Weight: 25% + 25% (acoustic+verbal) = 50%                         ║  
║                                                                       ║
║  Visual Output:                 Acoustic Output:                    ║
║  ├─ 7 emotion labels            ├─ Pitch (Hz)                      ║
║  ├─ Confidence scores           ├─ Energy (RMS)                    ║
║  ├─ Dominant emotion            ├─ Speech rate proxy               ║
║  ├─ Emotion timeline            ├─ Voice texture (MFCC)            ║
║  ├─ Head angles (°)             ├─ Sentiment label                 ║
║  └─ Eye contact %               ├─ 8 style scores                  ║
║                                  ├─ Filler word count              ║
║                                  └─ Clarity indicator               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow by Component

### Component 1: Facial Emotion Recognition

```
Video Frame (RGB)
    ↓
MediaPipe FaceMesh
    ↓ (468 landmarks)
Face ROI Extraction
    ↓ (crop)
224×224 resize + preprocessing
    ↓
ResNet-50 Backbone
    ├─ Extract 2048-dim features
    ├─ Feed to fully connected head
    └─ Output: 7-class logits
    ↓
LSTM Buffer (10-frame window)
    ├─ Input: 2048-dim features
    ├─ Hidden state tracking
    └─ Output: Smoothed 7-class logits
    ↓
Softmax + ArgMax
    ↓
EMOTION LABEL + CONFIDENCE
```

**Features Generated**:
- Per-frame emotion (label + score)
- Emotion segments (start/end/duration)
- Dominant emotion + overall confidence
- Emotion statistics (count/percentage/avg_confidence)

---

### Component 2: Head Pose & Gaze

```
Frame + MediaPipe Landmarks
    ↓
Select 6 canonical 3D points
  (nose, left eye, right eye, chin)
    ↓
Camera Matrix Setup
  (focal length ≈ frame width)
    ↓
OpenCV solvePnP()
    ├─ Input: 3D points + 2D image coords
    └─ Output: rvec (rotation vector)
    ↓
Extract Rotation Matrix
    ↓
RQDecomp3x3() → Euler angles
    ↓
OUTPUT:
├─ Pitch (up/down tilt)
├─ Yaw (left/right turn)
├─ Roll (head rotation)
└─ Eye Contact (|yaw| < 10° AND |pitch| < 10°)
```

---

### Component 3: Speech Transcription

```
Video → Extract Audio
    ↓ (moviepy)
PCM 16-bit, 16kHz, mono WAV
    ↓
Load as numpy float32
    ↓ (avoid ffmpeg call)
Whisper Model Preprocessing
    ├─ Convert to 80-dim MFCC spectrogram
    └─ Batch processing
    ↓
Whisper Encoder (12 layers)
    ↓
Whisper Decoder (12 layers, auto-regressive)
    ├─ Generate token stream
    └─ Per-token probability (confidence)
    ↓
Postprocess
    ├─ Timestamp alignment
    ├─ Segment grouping
    └─ Language detection
    ↓
OUTPUT:
├─ Full transcript (string)
├─ Language (auto-detected)
├─ Segment array
│  ├─ Start (sec)
│  ├─ End (sec)
│  ├─ Text (phrase)
│  └─ Confidence (0–1)
└─ Segment count
```

---

### Component 4: Acoustic Analysis

```
Audio WAV (16kHz)
    ↓
BRANCH 1: Pitch Extraction
    ├─ librosa.pyin()
    │  (Probabilistic YIN algorithm)
    ├─ F0 range: 65–2093 Hz
    ├─ Voiced/unvoiced classification
    └─ Statistics:
       ├─ Mean, Std, Min, Max, Range
       └─ Confidence indicator (high/mod/low)
    ↓
BRANCH 2: Energy
    ├─ RMS per frame
    ├─ Mean & variability
    └─ Energy level (high/mod/low)
    ↓
BRANCH 3: Speech Rate Proxy
    ├─ Zero-crossing rate
    ├─ Mean ZCR value
    └─ Articulation speed proxy
    ↓
BRANCH 4: Voice Texture
    ├─ 13 MFCC coefficients
    ├─ Mean across time
    └─ Voice signature
    ↓
BRANCH 5: Voiced Ratio
    ├─ % of audio with detected pitch
    └─ Speech vs. silence ratio
    ↓
OUTPUT:
├─ pitch { mean_hz, std_hz, min_hz, max_hz, range_hz }
├─ energy { mean, variability }
├─ speech_rate_proxy (float)
├─ voiced_ratio (percentage)
├─ mfcc_mean (13 values)
└─ interpretation {
     confidence_indicator,
     energy_level,
     pitch_variety
   }
```

---

### Component 5: NLP Analysis

```
Transcript (full_text)
    ↓
MAX TOKENS: 512 (Whisper/RoBERTa compatible)
    ↓
BRANCH 1: Sentiment (RoBERTa)
    ├─ Input: Text
    ├─ Model: cardiffnlp/twitter-roberta-base-sentiment-latest
    ├─ Output: 3 class scores
    │  ├─ NEGATIVE
    │  ├─ NEUTRAL
    │  └─ POSITIVE
    └─ Interpretation: positive/neutral/negative
    ↓
BRANCH 2: Communication Style (BART-MNLI)
    ├─ Input: Text
    ├─ Model: facebook/bart-large-mnli (zero-shot)
    ├─ Labels (8): confident, hesitant, enthusiastic, nervous,
    │              knowledgeable, vague, assertive, uncertain
    ├─ Output: Top-4 with scores
    └─ Multi-label (non-exclusive)
    ↓
BRANCH 3: Filler Word Detection
    ├─ Split text into words
    ├─ Filler word list:
    │  "um", "uh", "like", "basically", "literally",
    │  "you know", "so", "actually", "kind of", etc.
    ├─ Count matches (case-insensitive)
    └─ Metrics:
       ├─ Filler count
       ├─ Filler ratio (count / total_words)
       └─ Clarity indicator (high if ratio < 0.05)
    ↓
OUTPUT:
├─ sentiment {
│   label: "positive"|"neutral"|"negative",
│   score: 0–1
│ }
├─ communication_style {
│   confident: score,
│   enthusiastic: score,
│   knowledgeable: score,
│   ... (up to 8)
│ }
├─ word_count (integer)
├─ filler_word_count (integer)
├─ filler_ratio (0–1)
└─ clarity_indicator "high"|"moderate"|"low"
```

---

### Component 6: Composite Score & Fusion

```
emotion_analysis {
    dominant_emotion: str,
    overall_confidence: float,
    emotion_statistics: [{ emotion, count, %, avg_conf }]
}

gaze_analysis {
    eye_contact_percentage: float,
    average_pitch/yaw/roll: float
}

acoustic_analysis {
    pitch: { mean_hz, std_hz, ... },
    energy: { mean, variability },
    interpretation: { confidence_indicator, energy_level, pitch_variety }
}

nlp_analysis {
    sentiment: str,
    sentiment_score: float,
    communication_style: { label: score },
    filler_ratio: float,
    clarity_indicator: str
}
    ↓
SCORING LOGIC:
    ↓
1. Emotion Score (0–1)
   ├─ Positive emotions: Happiness, Surprise
   ├─ Negative emotions: Fear, Disgust, Anger, Sadness
   └─ Score = (Pos% × 0.6) + (1 − Neg% × 0.4)
    ↓
2. Gaze Score (0–1)
   └─ Score = eye_contact_pct / 100
    ↓
3. Acoustic Score (0–1)
   ├─ Pitch score = min(pitch_std / 30, 1.0)  [30 Hz = max]
   ├─ Energy score = min(energy_mean / 0.08, 1.0)  [0.08 RMS = max]
   └─ Score = (pitch_score + energy_score) / 2
    ↓
4. Verbal Score (0–1)
   ├─ Sentiment component = 1.0 if POSITIVE else 0.4
   ├─ Filler component = max(0, 1 − filler_ratio × 5)
   └─ Score = (sentiment × 0.5) + (filler × 0.5)
    ↓
5. Composite (0–100)
   └─ Score = 100 × (0.30×emotion + 0.20×gaze + 0.25×acoustic + 0.25×verbal)
    ↓
6. Grade Assignment
   ├─ 80–100 → "Excellent"
   ├─ 60–79 → "Good"
   ├─ 40–59 → "Average"
   └─ 0–39 → "Needs Review"
    ↓
7. Highlights Generation
   ├─ IF Happiness% > 30 → "Strong positive affect"
   ├─ IF eye_contact% > 70 → "Consistent eye contact"
   ├─ IF filler_ratio < 0.03 → "Clear speech"
   └─ IF pitch_std > 20 → "Varied vocal tone"
    ↓
OUTPUT:
├─ composite_score (0–100)
├─ grade (str)
├─ dimension_scores {
│   emotion: 0–100,
│   eye_contact: 0–100,
│   acoustic: 0–100,
│   verbal: 0–100
│ }
├─ weights_used { ... }
└─ highlights [str, str, str, str]
```

---

## Model Sizes & Memory Usage

| Component | Model Size | Loaded Size | Memory | Speed |
|-----------|-----------|------------|--------|-------|
| MediaPipe FaceMesh | 1 MB | 10 MB | Minimal | Real-time |
| ResNet-50 | 103 MB | 110 MB | ~200 MB | ~5ms/frame |
| LSTM | 45 MB | 50 MB | ~100 MB | ~2ms/frame |
| Whisper Base | 140 MB | 160 MB | ~300 MB | ~8–10 sec/min |
| RoBERTa | 498 MB | 520 MB | ~1 GB | <100ms |
| BART-MNLI | 1.63 GB | 1.8 GB | ~2 GB | <200ms |

**Total Footprint**: ~4–5 GB memory (all models loaded)  
**Recommendation**: GPU (CUDA) for 3–5× speedup

---

## Quick Reference: Endpoint → Response

### Request
```bash
curl -X POST "http://localhost:8000/analyze-with-transcription-new" \
  -F "file=@interview.mp4" \
  -F "candidate_name=Jane Smith" \
  -F "role_applied=ML Engineer"
```

### Response (JSON)
```json
{
  "success": true,
  "candidate": { ... },
  "candidate_score": {
    "composite_score": 78.5,
    "grade": "Good",
    "dimension_scores": {
      "emotion": 82.3,
      "eye_contact": 71.2,
      "acoustic": 76.8,
      "verbal": 75.4
    }
  },
  "emotion_analysis": { ... },
  "gaze_analysis": { ... },
  "acoustic_analysis": { ... },
  "verbal_analysis": { ... },
  "processing_time_seconds": 42.15
}
```

---

**Version**: 1.0  
**Last Updated**: 27 March 2026  
**All Components**: ✅ Implemented & Integrated
