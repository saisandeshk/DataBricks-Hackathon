"""Sarvam HTTP helpers: translation (Mayura), STT (Saaras), TTS (Bulbul).

Set ``SARVAM_API_KEY`` in the environment.
"""
from __future__ import annotations

import base64
import io
import os
import re
import wave
from typing import Any

import numpy as np
import requests

DEFAULT_TRANSLATE_URL = "https://api.sarvam.ai/translate"
DEFAULT_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_TTS_URL = "https://api.sarvam.ai/text-to-speech"


def get_api_key() -> str:
    return os.environ.get("SARVAM_API_KEY", "").strip()


def is_configured() -> bool:
    return bool(get_api_key())


def _subscription_headers(*, json_body: bool = True) -> dict[str, str]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set.")
    h: dict[str, str] = {"api-subscription-key": api_key}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


# ── Translation ──────────────────────────────────────────────────────────

def translate_text(
    text: str,
    *,
    source_language_code: str = "auto",
    target_language_code: str = "en-IN",
    timeout: int = 60,
) -> str:
    url = os.environ.get("SARVAM_TRANSLATE_URL", DEFAULT_TRANSLATE_URL).strip()
    body: dict[str, Any] = {
        "input": text,
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
    }
    r = requests.post(url, headers=_subscription_headers(), json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    for key in ("translated_text", "output", "text"):
        v = data.get(key)
        if v and str(v).strip():
            return str(v).strip()
    raise ValueError(f"Unexpected translate response: {data!r}")


# ── Speech-to-Text ───────────────────────────────────────────────────────

def speech_to_text_file(
    file_bytes: bytes,
    filename: str = "audio.wav",
    *,
    model: str | None = None,
    mode: str = "translate",
    language_code: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    url = os.environ.get("SARVAM_STT_URL", DEFAULT_STT_URL).strip()
    model = model or os.environ.get("SARVAM_STT_MODEL", "saaras:v3").strip()
    files = {"file": (filename, file_bytes, "audio/wav")}
    data: dict[str, str] = {"model": model, "mode": mode}
    if language_code:
        data["language_code"] = language_code
    r = requests.post(
        url, headers=_subscription_headers(json_body=False),
        files=files, data=data, timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def transcript_from_stt_response(resp: dict[str, Any]) -> str:
    t = resp.get("transcript")
    if t is None:
        raise ValueError(f"Unexpected STT response: {resp!r}")
    return str(t).strip()


# ── Text-to-Speech ───────────────────────────────────────────────────────

def text_to_speech_wav_bytes(
    text: str,
    *,
    target_language_code: str = "en-IN",
    speaker: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> bytes:
    url = os.environ.get("SARVAM_TTS_URL", DEFAULT_TTS_URL).strip()
    model = model or os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3").strip()
    body: dict[str, Any] = {
        "text": text[:2500],
        "target_language_code": target_language_code,
        "model": model,
    }
    if speaker:
        body["speaker"] = speaker
    r = requests.post(url, headers=_subscription_headers(), json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    audios = data.get("audios")
    if not audios:
        raise ValueError(f"Unexpected TTS response: {data!r}")
    return base64.b64decode(audios[0])


# ── Audio conversion helpers ─────────────────────────────────────────────

def wav_bytes_to_numpy_float32(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return sr, x


def numpy_audio_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    if samples is None or len(samples) == 0:
        raise ValueError("Empty audio")
    s = np.asarray(samples, dtype=np.float32)
    if s.ndim == 2:
        s = s.mean(axis=1)
    s = np.clip(s, -1.0, 1.0)
    pcm = (s * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def strip_markdown_for_tts(text: str, *, max_chars: int = 2400) -> str:
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]
