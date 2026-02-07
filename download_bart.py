"""
Download a BART zero-shot model for local/offline benchmarking and usage.

Usage:
  python services/topic-classifier-service/download_bart.py

This script downloads the `valhalla/distilbart-mnli-12-1` model by default and saves it
under `services/topic-classifier-service/models/valhalla-distilbart-mnli-12-1` using
the Transformers `save_pretrained` API so it can be loaded locally by the service.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = os.getenv("BART_MODEL", "valhalla/distilbart-mnli-12-1")
LOCAL_PATH = os.getenv("LOCAL_BART_PATH", "./models/valhalla-distilbart-mnli-12-1")


def download_bart():
    print(f"📦 Downloading {MODEL_NAME}...")
    print("   This may take a few minutes depending on your connection...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    os.makedirs(LOCAL_PATH, exist_ok=True)
    print(f"💾 Saving tokenizer and model to {LOCAL_PATH}...")
    tokenizer.save_pretrained(LOCAL_PATH)
    model.save_pretrained(LOCAL_PATH)
    print("✅ Model saved successfully")


if __name__ == "__main__":
    download_bart()
