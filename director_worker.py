"""Director worker — separate process running llama.cpp on GPU (n_gpu_layers=-1).

Dedicated CUDA context and its own cuBLAS 12.4 (not shared with torch) -> "CUDA error: invalid argument"
is prevented by design. Reads ONE JSON request from stdin, prints JSON result to stdout,
and terminates (freeing VRAM before TTS). IMPORTANT: torch is NOT imported here — otherwise
its cuBLAS 12.8 would be loaded into the process and cause symbol collisions.
"""
import os
import sys
import json
import re
import traceback

# Pinned memory in llama causes einval with pinned allocator — disabled (does not affect layer offloading).
os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")
# CUDA_VISIBLE_DEVICES is untouched — director runs on GPU.

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

MODELS = {
    "Qwen3.5-9B · Q4_K_M (default, ~5.5 GB)": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    "Qwen3.5-4B · Q4_K_M (light, ~2.5 GB)": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
}
DEFAULT_MODEL = "Qwen3.5-9B · Q4_K_M (default, ~5.5 GB)"


def log(msg):
    print(f"[director] {msg}", file=sys.stderr, flush=True)


def filter_tags(text):
    if not text:
        return text

    def repl(m):
        s = m.group(0)
        v = _VALID.fullmatch(s)
        if v and v.group(2) in WHITELIST.get(v.group(1), ()):
            return s
        return ""

    return _ANGLE.sub(repl, text)


def load_llm(label=DEFAULT_MODEL):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    entry = MODELS.get(label)
    if not entry:
        for k, v in MODELS.items():
            if ("9B" in label and "9B" in k) or ("4B" in label and "4B" in k):
                entry = v
                break
        if not entry:
            entry = MODELS[DEFAULT_MODEL]
    repo, fname = entry
    log(f"loading {label} on GPU...")
    path = hf_hub_download(repo, fname)
    return Llama(model_path=path, n_gpu_layers=-1, n_ctx=8192, verbose=False)


def _strip_think(txt):
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()


def _chat(llm, system, user, max_new=1024, temp=0.4):
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_new, temperature=temp, top_p=0.9)
    return _strip_think(out["choices"][0]["message"].get("content", ""))


_TAG_RULES = (
    "ONLY these tags are allowed, strictly in the format <|category:value|> (with vertical bars):\n"
    "- emotion (at the beginning of a sentence): " + ", ".join(sorted(WHITELIST["emotion"])) + "\n"
    "- prosody: " + ", ".join(sorted(WHITELIST["prosody"])) + " (pause/long_pause — inside the line)\n"
    "- style (at the beginning of a sentence): " + ", ".join(sorted(WHITELIST["style"])) + "\n"
    "- sfx (inside the line, immediately adjacent to the onomatopoeia/sound): " + ", ".join(sorted(WHITELIST["sfx"])) + "\n"
    "DO NOT invent other tags or values. It is FORBIDDEN to write <speed_1.2>, <emotion:excited>, <sfx:wind> — "
    "only values from the list and only in <|category:value|> format.\n"
    "Example: <|emotion:elation|>Congratulations everyone! <|sfx:laughter|>haha. <|prosody:long_pause|> Continuing.\n"
    "Return ONLY the resulting text, without explanations or preamble."
)


def _filter_line(line):
    """Keep 'NAME:' prefix, filter tags in the spoken part."""
    if ":" in line:
        who, _, said = line.partition(":")
        return f"{who}:{filter_tags(said)}"
    return filter_tags(line)


def enrich(llm, text):
    s = ("You are a voice director. Keep the original language of the text. Normalize the text for pronunciation "
         "(numbers, dates, abbreviations, currencies, units, symbols as words), fix obvious typos, and place "
         "emotional / sfx / prosody tags according to meaning. " + _TAG_RULES)
    return filter_tags(_chat(llm, s, text))


def write_podcast(llm, topic, n_speakers=2):
    n = max(2, int(n_speakers))
    s = (f"You are a podcast scriptwriter for {n} speakers (Speaker 0 .. Speaker {n - 1}). Match the language of the prompt. "
         "Write an engaging dialogue: EVERY line strictly in the format 'Speaker K: line', where K is a number from 0. "
         "Give each speaker their own speaking style and personality. Place tags according to meaning in dialogue. " + _TAG_RULES)
    out = _chat(llm, s, topic, max_new=2048)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def cast_audiobook(llm, text, n_voices=2):
    n = max(2, int(n_voices))
    s = ("You are an audiobook casting director. Keep the original text language. Separate the text into narrator speech and character dialogue. "
         f"Speaker 0 is the NARRATOR (author text), Speaker 1 .. Speaker {n - 1} are characters "
         "(assign each character a persistent speaker number and maintain it consistently). "
         "EVERY line must be strictly in the format 'Speaker K: line'. Keep the text VERBATIM, "
         "only marking the speaker and adding tags according to meaning. " + _TAG_RULES)
    out = _chat(llm, s, text, max_new=2048)
    return "\n".join(_filter_line(ln) for ln in out.splitlines())


def main():
    try:  # UTF-8 on pipe regardless of environment (Windows pipe is otherwise ANSI)
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad request: {e}"}, ensure_ascii=False))
        return
    action = req.get("action")
    text = req.get("text", "")
    label = req.get("label", DEFAULT_MODEL)
    n = req.get("n", 2)
    try:
        llm = load_llm(label)
        if action == "enrich":
            result = enrich(llm, text)
        elif action == "podcast":
            result = write_podcast(llm, text, n)
        elif action == "audiobook":
            result = cast_audiobook(llm, text, n)
        else:
            result = text
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        sys.stdout.flush()
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
