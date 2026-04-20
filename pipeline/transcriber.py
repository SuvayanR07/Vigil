"""
Local speech-to-text for audio adverse-event reports.

Uses OpenAI Whisper (runs locally on CPU, no API key). The "base" model
is ~140 MB and is downloaded on first use and cached thereafter.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


class TranscriberDependencyError(RuntimeError):
    """Raised when Whisper is not installed or cannot be loaded."""


# Module-level cache so the model is loaded only once per process
_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = "base"


def _load_whisper():
    """Lazy-import whisper and memoize the model instance."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise TranscriberDependencyError(
            "Audio transcription requires `openai-whisper`. Install with:\n"
            "  pip install openai-whisper\n"
            "ffmpeg is also required:\n"
            "  macOS : brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg"
        ) from e

    try:
        _WHISPER_MODEL = whisper.load_model(_WHISPER_MODEL_NAME)
    except Exception as e:
        raise TranscriberDependencyError(
            f"Could not load Whisper '{_WHISPER_MODEL_NAME}' model: {e}"
        ) from e

    return _WHISPER_MODEL


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        file_bytes: Raw audio bytes (mp3 / wav / m4a / ogg).
        filename:   Original filename — its extension is preserved for the
                    temp file so Whisper/ffmpeg can infer the codec.

    Returns:
        Transcribed text (stripped).

    Raises:
        TranscriberDependencyError: if Whisper or ffmpeg is missing.
        ValueError: if transcription fails.
    """
    model = _load_whisper()

    # Preserve original extension so ffmpeg picks the right demuxer
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".wav"

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path, fp16=False)
        except Exception as e:
            raise ValueError(f"Whisper transcription failed: {e}") from e

        text = (result.get("text") or "").strip()
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def is_available() -> tuple[bool, Optional[str]]:
    """Probe whether audio transcription is ready."""
    try:
        import whisper  # noqa: F401
        return True, None
    except ImportError as e:
        return False, (
            "openai-whisper is not installed. Run `pip install openai-whisper` "
            "and ensure ffmpeg is on PATH."
        )
    except Exception as e:
        return False, str(e)
