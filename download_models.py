"""
Download ONNX models for topic classification
Run this script once to download models locally for faster Docker builds
"""

import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

# Model configuration
MODEL_REPO = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
LOCAL_MODEL_DIR = Path("./models/deberta-onnx")
ONNX_SUBFOLDER = LOCAL_MODEL_DIR / "onnx"

def download_model():
    print(f"📦 Downloading {MODEL_REPO}...")
    print(f"   This may take 2-3 minutes...")

    # Create directories
    os.makedirs(ONNX_SUBFOLDER, exist_ok=True)

    # Temporarily disable offline mode to download
    os.environ["HF_HUB_OFFLINE"] = "0"

    # Files to download
    files = {
        "config.json": LOCAL_MODEL_DIR,
        "tokenizer.json": LOCAL_MODEL_DIR,
        "tokenizer_config.json": LOCAL_MODEL_DIR,
        "special_tokens_map.json": LOCAL_MODEL_DIR,
        "spm.model": LOCAL_MODEL_DIR,
        "onnx/model.onnx": ONNX_SUBFOLDER,
    }

    for filename, folder in files.items():
        print(f"  📥 Downloading {filename}...")
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                repo_type="model"
            )
            shutil.copy(path, folder / Path(filename).name)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not download {filename}: {e}")

    print(f"\n✅ Model saved successfully!")
    print(f"   Location: {LOCAL_MODEL_DIR.absolute()}")
    print(f"   Files: {list(LOCAL_MODEL_DIR.rglob('*'))}")


if __name__ == "__main__":
    download_model()
