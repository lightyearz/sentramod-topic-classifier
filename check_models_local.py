"""
Quick script to test whether ONNX and BART models can be loaded from the local filesystem
without attempting network downloads.

Usage:
  python services/topic-classifier-service/check_models_local.py

This will attempt to load the onnx model at settings.ONNX_MODEL and a local BART model if present.
It uses local_files_only=True to confirm there are no HF network downloads in the process.
"""

import os
import json
import sys
from transformers import AutoTokenizer, pipeline

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
except Exception as e:
    print(
        "optimum.onnxruntime import failed; please make sure 'optimum' is installed. Error:",
        e,
    )
    ORTModelForSequenceClassification = None

script_dir = os.path.dirname(__file__)
# Insert the service app directory into sys.path so we can import app.config
service_app_dir = os.path.abspath(os.path.join(script_dir, "app"))
if service_app_dir not in sys.path:
    sys.path.insert(0, service_app_dir)
try:
    from config import settings  # config.py in services/topic-classifier-service/app
except Exception:
    print("Could not import settings from app.config, falling back to defaults.")

    class Dummy:
        ONNX_MODEL = "./services/topic-classifier-service/models/deberta-onnx"
        BART_MODEL = "valhalla/distilbart-mnli-12-1"

    settings = Dummy()


def test_onnx_model(path: str):
    print("\n-- ONNX Model Test --")
    print(f"Attempting to load ONNX model from: {path}")
    if not os.path.exists(path):
        print("Path does not exist:", path)
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        print(
            "Tokenizer loaded (local_files_only=True).",
            "fast" if getattr(tokenizer, "is_fast", False) else "slow",
        )
    except Exception as e:
        print("Tokenizer failed to load for ONNX path:", e)
        tokenizer = None

    if ORTModelForSequenceClassification is None:
        print("optimum ONNX model class not available, skipping ONNX model load test")
        return

    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        )
        print("ORTModel loaded (local_files_only=True)")
    except Exception as e:
        print("ORTModel failed to load (local_files_only=True):", e)
        model = None

    if tokenizer is None or model is None:
        print("ONNX model or tokenizer not available for inference test")
        return

    try:
        classifier = pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer, device=-1
        )
        print("Pipeline created successfully. Running a dry sample...")
        out = classifier(
            "I am worried about my friend",
            ["suicide", "self-harm", "anger"],
            multi_label=True,
        )
        print("Output (truncated):")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("Pipeline creation or inference failed:", e)


def test_bart_model(path: str):
    print("\n-- BART Model Test --")
    print(f"Attempting to load BART model from: {path} (local_files_only=True)")
    try:
        # This will only work if the local files exist; it will not download
        classifier = pipeline(
            "zero-shot-classification", model=path, device=-1, local_files_only=True
        )
        print(
            "BART pipeline created successfully (local_files_only=True). Running a dry sample..."
        )
        out = classifier(
            "I am worried about my friend",
            ["suicide", "self-harm", "anger"],
            multi_label=True,
        )
        print("Output (truncated):")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("BART pipeline failed to load or run locally:", e)


if __name__ == "__main__":
    onnx_path = getattr(
        settings,
        "ONNX_MODEL",
        "./services/topic-classifier-service/models/deberta-onnx",
    )
    bart_model = getattr(settings, "BART_MODEL", "valhalla/distilbart-mnli-12-1")

    print("Settings:")
    print(" ONNX_MODEL:", onnx_path)
    print(" BART_MODEL:", bart_model)

    test_onnx_model(onnx_path)
    test_bart_model(bart_model)
