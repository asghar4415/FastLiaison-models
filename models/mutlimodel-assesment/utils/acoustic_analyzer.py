import librosa
import numpy as np


def analyze_acoustics(audio_path: str, sr: int = 16000) -> dict:
    """
    Extract paralinguistic acoustic features from audio file.
    Maps to paper Section 3.2: pitch, loudness, speech rate, intonation.
    
    Args:
        audio_path: Path to WAV audio file
        sr: Sample rate (default 16000 to match Whisper extraction)
        
    Returns:
        Dictionary of acoustic features with interpretations
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        if len(y) == 0:
            return {"error": "Empty audio file"}

        # --- Pitch (F0) via pyin ---
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz — lower male voice floor
            fmax=librosa.note_to_hz('C7'),   # ~2093 Hz — upper female voice ceiling
            sr=sr
        )
        f0_clean = f0[voiced_flag]  # only voiced frames

        # Safe scalar extraction
        f0_mean  = float(np.nanmean(f0_clean))  if len(f0_clean) > 0 else 0.0
        f0_std   = float(np.nanstd(f0_clean))   if len(f0_clean) > 0 else 0.0
        f0_min   = float(np.nanmin(f0_clean))   if len(f0_clean) > 0 else 0.0
        f0_max   = float(np.nanmax(f0_clean))   if len(f0_clean) > 0 else 0.0
        f0_range = f0_max - f0_min

        # --- Energy / loudness (RMS) ---
        rms         = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))

        # --- Speech rate proxy (zero crossing rate — higher = faster articulation) ---
        zcr              = librosa.feature.zero_crossing_rate(y)[0]
        speech_rate_proxy = float(np.mean(zcr))

        # --- MFCCs (voice texture / timbre) ---
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = [round(float(m), 4) for m in np.mean(mfccs, axis=1)]

        # --- Voiced ratio (what % of audio has speech) ---
        voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.0

        # --- Interpretations ---
        # Confidence indicator: higher mean pitch + more pitch variation = more engaged/confident speaker
        confidence_indicator = "high"   if f0_mean > 150 and f0_std > 15 else \
                               "moderate" if f0_mean > 100 else "low"

        # Energy level: RMS proxy for loudness / projection
        energy_level = "high"   if energy_mean > 0.05 else \
                       "moderate" if energy_mean > 0.02 else "low"

        # Pitch variety: monotone vs expressive
        pitch_variety = "expressive" if f0_std > 25 else \
                        "moderate"   if f0_std > 10 else "monotone"

        return {
            "pitch": {
                "mean_hz":  round(f0_mean,  2),
                "std_hz":   round(f0_std,   2),
                "min_hz":   round(f0_min,   2),
                "max_hz":   round(f0_max,   2),
                "range_hz": round(f0_range, 2),
            },
            "energy": {
                "mean":        round(energy_mean, 5),
                "variability": round(energy_std,  5),
            },
            "speech_rate_proxy": round(speech_rate_proxy, 5),
            "voiced_ratio":      round(voiced_ratio, 3),
            "mfcc_mean":         mfcc_mean,
            "interpretation": {
                "confidence_indicator": confidence_indicator,
                "energy_level":         energy_level,
                "pitch_variety":        pitch_variety,
            }
        }

    except Exception as e:
        return {"error": str(e)}
    