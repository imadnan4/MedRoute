#!/usr/bin/env python
"""Download the Nemotron 3.5 ASR ONNX INT4 model for local inference.

The model is ~300MB and runs entirely on CPU. Downloaded once, cached forever.

Usage:
    python scripts/download_asr_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice.nemotron_local import _ensure_model, DEFAULT_MODEL_DIR

print("Downloading Nemotron 3.5 ASR ONNX INT4 model (~300MB)...")
print("This is a one-time download. The model runs on CPU - no GPU needed.")
print()

path = _ensure_model(DEFAULT_MODEL_DIR)

print()
print(f"Model ready at {path}")
print("You can now use voice transcription locally.")
print("Try: curl -X POST http://localhost:8000/transcribe")
