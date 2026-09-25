"""AI text director — runs in a SEPARATE GPU process (isolating the CUDA runtime from torch).

Why a separate process instead of in-process: llama.cpp (cu124 build) and torch (cu126/cu128)
in the same process share cublas64_12.dll by NAME — Windows keeps one loaded module per process,
and conflicting versions cause "CUDA error: invalid argument". Matching builds are not feasible
(abetlen's llama wheels max out at cu124, while torch 2.7.1 lacks cu124).
Therefore, the director runs in its own process with its own cuBLAS 12.4.
RUNS ON GPU (n_gpu_layers=-1) — separate process != CPU; full speed is retained. The worker
is short-lived: it loads the model, generates the response, and exits, releasing VRAM before TTS.

filter_tags / WHITELIST / MODELS — pure Python (no llama/GPU), used by UI and tests.
"""
import os
import re
import json
import subprocess
import sys
import threading

_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))

WHITELIST = {
    "emotion": {"affection", "amusement", "anger", "arousal", "awe", "bitterness", "confusion",
                "contemplation", "contentment", "determination", "disgust", "elation", "enthusiasm",
                "fear", "helplessness", "longing", "pride", "relief", "sadness", "shame", "surprise"},
    "prosody": {"speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pitch_low",
                "pitch_high", "expressive_high", "expressive_low", "pause", "long_pause"},
    "style": {"singing", "shouting", "whispering"},
    "sfx": {"cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"},
}

_ANGLE = re.compile(r"<[^<>\n]{1,48}>")
_VALID = re.compile(r"<\|(\w+):(\w+)\|>")


def filter_tags(text):
    """Keep ONLY valid <|cat:val|> from whitelist; strip other angle-bracket patterns."""
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""

    return _ANGLE.sub(repl, text)


MODELS = {  # GGUF (llama.cpp, GPU): downloads 4-bit Q4_K_M
    "Qwen3.5-9B · Q4_K_M (default, ~5.5 GB)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (light, ~2.5 GB)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (default, ~5.5 GB)"

_lock = threading.Lock()


def _call(action, text, label=DEFAULT_MODEL, n=2):
    """Launch director worker (separate GPU process), send request, receive result.
    Worker terminates on its own -> its VRAM and CUDA context are freed before TTS starts."""
    if _MOCK:
        return text
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "director_worker.py")
    req = json.dumps({"action": action, "text": text, "label": label, "n": n}, ensure_ascii=False)
    # GPU is NOT disabled — director runs on GPU. This process contains no torch; llama uses its own cuBLAS 12.4.
    # stderr -> console (inherited): logs and download progress are visible, pipe buffer does not fill up.
    with _lock:  # one worker at a time — avoid loading two models on GPU concurrently
        proc = subprocess.run(
            [sys.executable, script],
            input=req, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=os.environ.copy(),
        )
    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(f"director worker did not respond (code {proc.returncode}) — see console log")
    data = json.loads(out.splitlines()[-1])  # last stdout line is the JSON response
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "director: unknown error"))
    return data["result"]


def enrich(text, label=DEFAULT_MODEL):
    """ROLE A — normalization for pronunciation + light cleanup + tags according to meaning."""
    return _call("enrich", text, label)


def write_podcast(topic, n_speakers=2, label=DEFAULT_MODEL):
    """ROLE B — multi-speaker dialogue in indexed format 'Speaker N: line'."""
    return _call("podcast", topic, label, n=max(2, int(n_speakers)))


def cast_audiobook(text, n_voices=2, label=DEFAULT_MODEL):
    """ROLE C — attribution: Speaker 0 = narrator, 1.. = characters."""
    return _call("audiobook", text, label, n=max(2, int(n_voices)))
