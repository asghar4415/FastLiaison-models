"""
One-time download script for NLP models.
Run this ONCE to save models locally into /models directory alongside your .pt files.

Usage:
    python download_nlp_models.py

Models downloaded:
    - cardiffnlp/twitter-roberta-base-sentiment-latest  (~500 MB)
    - facebook/bart-large-mnli                          (~1.6 GB)

After running, your /models directory will look like:
    models/
        FER_static_ResNet50_AffectNet.pt        (existing)
        FER_dinamic_LSTM_Aff-Wild2.pt           (existing)
        nlp/
            sentiment/                          (NEW)
                config.json
                tokenizer_config.json
                vocab.json
                merges.txt
                model.safetensors
                ...
            zero_shot/                          (NEW)
                config.json
                tokenizer_config.json
                vocab.json
                model.safetensors
                ...
"""

import os
import sys
from pathlib import Path

# IMPORTANT: Set HuggingFace cache to local directory BEFORE importing transformers
# This prevents downloads to C:\Users\...\cache\huggingface
MODELS_DIR = Path(__file__).parent / "models" / "nlp"
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HF_DATASETS_CACHE"] = str(MODELS_DIR / "datasets")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)


def download_model(repo_id: str, save_path: str, model_name: str):
    """Download and save a HuggingFace model + tokenizer locally."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Skip if already downloaded (check for config.json as marker)
    if (save_path / "config.json").exists():
        print(f"  ✓ {model_name} already exists at {save_path} — skipping download")
        return True
    
    print(f"  ⬇ Downloading {model_name} from {repo_id} ...")
    print(f"  → Saving to: {save_path}")
    
    try:
        # Download tokenizer
        print(f"    Fetching tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=str(save_path),
            force_download=False,
            resume_download=True
        )
        tokenizer.save_pretrained(str(save_path))
        
        # Download model weights
        print(f"    Fetching model weights (this may take a while)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            cache_dir=str(save_path),
            force_download=False,
            resume_download=True
        )
        model.save_pretrained(str(save_path))
        
        print(f"  ✓ {model_name} saved successfully!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download {model_name}: {e}")
        return False


def main():
    # Resolve /models path relative to this script's location
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models" / "nlp"
    
    print("=" * 60)
    print("MMIA — NLP Model Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {models_dir.resolve()}")
    print(f"\nModels to download:")
    print(f"  1. cardiffnlp/twitter-roberta-base-sentiment-latest (~500 MB)")
    print(f"  2. facebook/bart-large-mnli (~1.6 GB)")
    print(f"\nTotal disk space needed: ~2.1 GB")
    print("=" * 60)
    
    # Confirm
    answer = input("\nProceed with download? (y/n): ").strip().lower()
    if answer != 'y':
        print("Aborted.")
        sys.exit(0)
    
    print()
    
    results = {}
    
    # 1. Sentiment model
    results["sentiment"] = download_model(
        repo_id   = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        save_path = models_dir / "sentiment",
        model_name= "Sentiment (RoBERTa)"
    )
    
    # 2. Zero-shot classifier
    results["zero_shot"] = download_model(
        repo_id   = "facebook/bart-large-mnli",
        save_path = models_dir / "zero_shot",
        model_name= "Zero-Shot Classifier (BART-MNLI)"
    )
    
    # Summary
    print("=" * 60)
    print("Download Summary:")
    all_ok = True
    for name, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {status}  {name}")
        if not success:
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("\n✅ All models ready. You can now run the API server.")
        print("   NLP models will load from local disk — no internet needed.")
    else:
        print("\n⚠  Some models failed. Check your internet connection and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()