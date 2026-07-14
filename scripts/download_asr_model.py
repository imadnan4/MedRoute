#!/usr/bin/env python
"""Download and initialize the configured faster-whisper ASR model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings
from voice.whisper_local import get_local_asr

print(f"Downloading Whisper ASR model: {settings.asr_model}")
print(f"Cache directory: {settings.asr_cache_dir}")
print("The first download is large; subsequent starts use the local cache.")
print()

get_local_asr().ensure_model()

print()
print("Whisper ASR is ready.")
print("Use the browser voice recorder or POST audio to /transcribe.")
