"""Higgs Audio v3 TTS Engine.

Loads transformers port multimodalart/higgs-audio-v3-tts-4b-transformers,
auto precision (bf16/8/4-bit via bitsandbytes according to VRAM), generate_speech,
long-form with voice carry-over, and multi-speaker concatenation.

Heavy imports are lazy (inside functions) — mock mode and UI can start without torch.
"""
import os

TTS_REPO = "multimodalart/higgs-audio-v3-tts-4b-transformers"
SR = 24000
_MOCK = bool(os.environ.get("HIGGS_UI_MOCK"))
_model = None
_tok = None

# Cloning: reference is encoded IN FULL. Presets can be up to 300s -> prefill with thousands of audio tokens
# -> latency/stalling/degraded audio; affects both podcast and audiobook (shared generate path). We trim it.
REF_MAX_SEC = 30
_REF_CACHE = {}  # path -> (mtime, codes_TN, trimmed): avoid re-encoding the same voice multiple times


def detect_device():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "cuda", p.name, p.total_memory / 1e9
    return "cpu", "CPU", 0.0


def device_info():
    if _MOCK:
        return "MOCK UI (no model)"
    try:
        dev, name, vram = detect_device()
    except Exception:
        return "CPU"
    return f"{name} | VRAM {vram:.1f} GB" if dev == "cuda" else "CPU (slow)"


_forced_precision = None  # UI quantization choice


def set_precision(p):
    """UI quantization choice: '4bit' / '8bit' / 'bf16'. Unloads model — will reload in new precision."""
    global _forced_precision
    _forced_precision = p if p in ("4bit", "8bit", "bf16") else None
    unload_tts()


def auto_precision(vram_gb, device):
    # bf16 by default (cleaner; fits easily on 24 GB). nf4/8bit via UI selection.
    # Priority: UI choice > env HIGGS_TTS_PRECISION > default.
    pick = _forced_precision or os.environ.get("HIGGS_TTS_PRECISION", "").strip().lower()
    if pick in ("bf16", "8bit", "4bit"):
        return pick if device == "cuda" else "cpu"
    if device == "cpu":
        return "cpu"
    return "bf16"


def get_tts(precision=None):
    global _model, _tok
    if _MOCK:
        return "MOCK"
    if _model is not None:
        return _model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device, name, vram = detect_device()
    precision = precision or auto_precision(vram, device)
    print(f"[higgs] loading TTS ({precision}) on {name}...")
    quant = None
    # Keep audio code head and embedding OUTSIDE quantization: they are tied and predict stop tokens (EOC);
    # under nf4 the head degrades -> model misses the end -> generation runs away / stalls.
    skip = ["audio_head", "audio_embedding"]
    if precision == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                   llm_int8_skip_modules=skip)
    elif precision == "8bit":
        quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    _tok = AutoTokenizer.from_pretrained(TTS_REPO)
    kw = dict(trust_remote_code=True, dtype=torch.bfloat16)
    if quant is not None:
        kw["quantization_config"] = quant
        kw["device_map"] = "auto"
    try:  # flash-attention 2 if installed (accelerator from install.bat)
        _model = AutoModelForCausalLM.from_pretrained(TTS_REPO, attn_implementation="flash_attention_2", **kw)
    except Exception as e:
        print(f"[higgs] flash_attention_2 unavailable ({e}); using standard attention")
        _model = AutoModelForCausalLM.from_pretrained(TTS_REPO, **kw)
    if quant is None and device == "cuda":
        _model = _model.to("cuda")
    _model.eval()
    try:
        _model.get_audio_codec()  # warm up fp32 codec
    except Exception:
        pass
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")  # TF32 — safe and free
        except Exception:
            pass
    # torch.compile backbone: ~2.4x (dynamic=True). CRITICAL: compilation must happen in the MAIN
    # thread (startup warmup, see prewarm() in app.py) — dynamo/inductor compilation in gradio worker
    # thread crashes process (especially on clone path). Requires Python headers (install.bat installs dev.msi).
    if device == "cuda" and precision == "bf16" and os.environ.get("HIGGS_NO_COMPILE", "").lower() not in ("1", "true", "yes"):
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True  # compilation failure in thread -> fallback to eager, NOT crash
            _model.model = torch.compile(_model.model, dynamic=True)
            print("[higgs] torch.compile (dynamic) ON — ~2x (warmup compilation on startup)")
        except Exception as e:
            print(f"[higgs] torch.compile unavailable ({e}); running uncompiled")
    return _model


_CANCEL = False


def request_cancel():
    global _CANCEL
    _CANCEL = True
    print("[gen] STOP CANCEL — aborting generation at current token", flush=True)


def clear_cancel():
    global _CANCEL
    _CANCEL = False


def cancelled():
    return _CANCEL


def unload_tts():
    """Unload TTS from memory (for sequential loading with LLM director)."""
    global _model, _tok
    _model = None
    _tok = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_ref(path):
    import torch
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data).mean(dim=1), sr  # mono [L], sr


def _ref_codes(m, path):
    """Reference -> codes [T,N]: trim to REF_MAX_SEC + cache by (path, mtime).
    Returns (codes_cpu, trimmed). Without trimming, long presets (up to 300s) are encoded in full ->
    thousands of prefill tokens -> voice cloning slows down/stalls (affecting podcast and audiobook)."""
    import os
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0.0
    hit = _REF_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1], hit[2]
    wav, sr = _load_ref(path)
    cap = int(sr * REF_MAX_SEC)
    trimmed = wav.shape[-1] > cap
    if trimmed:
        wav = wav[:cap]
        print(f"[gen] reference {os.path.basename(path)} trimmed to {REF_MAX_SEC}s (was too long)", flush=True)
    codes = m._encode_reference(wav, sr).cpu()
    _REF_CACHE[path] = (mt, codes, trimmed)
    return codes, trimmed


_FRAMES_PER_SEC = None  # calibrated after first generation — for on-the-fly audio seconds estimation


def _modeling(m):
    """Model remote-code module (apply_delay_pattern / reverse_delay_pattern / _SamplerState / _sampler_step).
    torch.compile wraps m.model, not m itself — so type(m).__module__ remains the Higgs module."""
    import sys
    return sys.modules[type(m).__module__]


def _generate_stream(m, tok, text, *, reference_audio=None, reference_sample_rate=None,
                     reference_codes=None, reference_text=None, max_new_tokens=2048,
                     temperature=1.0, top_p=None, top_k=None, label="", attempt=(1, 1)):
    """Exact replica of HiggsMultimodalQwen3.generate_speech, but with custom loop ->
    (a) live per-frame progress in terminal (tqdm); (b) cancellation AT INFERENCE LEVEL —
    _CANCEL flag is checked every token to break the actual AR loop immediately.

    Returns (audio_tensor_cpu_f32 | empty, was_cancelled: bool).
    """
    global _FRAMES_PER_SEC
    import time
    import torch

    # Resolve model internals; if remote API changes — fallback to standard method.
    try:
        mod = _modeling(m)
        apply_delay_pattern = mod.apply_delay_pattern
        reverse_delay_pattern = mod.reverse_delay_pattern
        _SamplerState = mod._SamplerState
        _sampler_step = mod._sampler_step
        N = m.num_codebooks
        _ = (m._encode_reference, m._build_prompt_ids, m._prefill_embeds, m._decode_codes,
             m.audio_head, m.audio_embedding, m.model)
    except Exception as e:
        print(f"[gen] internal loop unavailable ({e}); falling back to default generate_speech without progress", flush=True)
        kw = dict(reference_text=reference_text, max_new_tokens=max_new_tokens,
                  temperature=temperature, top_p=top_p, top_k=top_k)
        if reference_codes is not None:
            kw["reference_codes"] = reference_codes
        elif reference_audio is not None:
            kw["reference_audio"] = reference_audio
            kw["reference_sample_rate"] = reference_sample_rate
        return m.generate_speech(text, tok, **kw), False

    att = f" · attempt {attempt[0]}/{attempt[1]}" if attempt[1] > 1 else ""
    print(f"[gen] >> synthesis: {label}{att}", flush=True)

    with torch.no_grad():
        delayed_ref = None
        if reference_codes is not None:
            delayed_ref = apply_delay_pattern(reference_codes.to(torch.long))
        elif reference_audio is not None:
            sr = reference_sample_rate or m.config.sample_rate
            codes_TN = m._encode_reference(reference_audio, sr)
            delayed_ref = apply_delay_pattern(codes_TN.cpu())

        prompt_ids = m._build_prompt_ids(
            tok, text,
            num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
            reference_text=reference_text,
        )
        inputs_embeds = m._prefill_embeds(prompt_ids, delayed_ref)
        out = m.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

        state = _SamplerState(num_codebooks=N)
        rows = []
        cancelled_mid = False
        t0 = time.time()
        try:
            from tqdm import tqdm
            bar = tqdm(total=int(max_new_tokens), unit="frame", desc="[gen] tts",
                       dynamic_ncols=True, leave=False, ascii=True)
        except Exception:
            bar = None

        for step in range(int(max_new_tokens)):
            if _CANCEL:  # <-- CANCELLATION AT INFERENCE LEVEL: interrupt actual generation loop
                cancelled_mid = True
                break
            logits_NV = m.audio_head(hidden_last).to(torch.float32)[0]
            codes_N = _sampler_step(logits_NV, state, temperature=temperature, top_p=top_p, top_k=top_k)
            if state.generation_done:
                break
            rows.append(codes_N.cpu())

            if bar is not None:
                bar.update(1)
                if _FRAMES_PER_SEC and step % 16 == 0:
                    bar.set_postfix_str(f"~{len(rows) / _FRAMES_PER_SEC:.1f}s audio")
            elif step % 64 == 0 and step:
                el = time.time() - t0
                print(f"[gen]   {step} frames · {step / max(el, 1e-3):.0f} frames/s", flush=True)

            step_embed = m.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
            cache_pos = torch.tensor([position], device=m.device)
            out = m.model(inputs_embeds=step_embed.to(inputs_embeds.dtype),
                          past_key_values=past, use_cache=True, cache_position=cache_pos)
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position += 1

        if bar is not None:
            bar.close()
        el = time.time() - t0

        if cancelled_mid:
            print(f"[gen] STOP interrupted at {len(rows)} frames ({el:.1f}s)", flush=True)
            return torch.zeros(0, dtype=torch.float32), True
        if len(rows) < N:
            print(f"[gen] empty ({len(rows)} frames < {N})", flush=True)
            return torch.zeros(0, dtype=torch.float32), False

        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        audio = m._decode_codes(codes_TN)

    sec = audio.shape[-1] / SR
    if sec > 0.05:
        _FRAMES_PER_SEC = len(rows) / sec  # calibrate seconds estimation for subsequent generations
    print(f"[gen] OK {len(rows)} frames -> {sec:.1f}s audio in {el:.1f}s ({len(rows) / max(el, 1e-3):.0f} frames/s)",
          flush=True)
    return audio, False


def generate(text, ref_audio=None, ref_text=None, temperature=1.0, top_p=0.95,
             top_k=50, max_new_tokens=2048, seed=-1):
    """Synthesize one fragment. Returns (sr, np.float32[L]). ref_audio — path to audio file."""
    import numpy as np
    text = (text or "").strip()
    if not text:
        return SR, np.zeros(0, np.float32)
    if _MOCK:
        n = int(SR * 1.4)
        return SR, (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)
    import torch
    m = get_tts()
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    kw = dict(max_new_tokens=int(max_new_tokens), temperature=max(0.05, float(temperature)),
              top_p=(float(top_p) if float(top_p) < 1.0 else None),
              top_k=(int(top_k) if int(top_k) > 0 else None))
    if ref_audio:
        ref_codes, trimmed = _ref_codes(m, ref_audio)
        kw["reference_codes"] = ref_codes
        if trimmed:
            ref_text = None  # preset transcript belongs to FULL audio -> will not match trimmed audio
        if ref_text and ref_text.strip():
            kw["reference_text"] = ref_text.strip()
    label = f"{len(text)} chars" + (" · voice clone from reference" if ref_audio else "")
    # Anti-runaway (like retry_badcase in VoxCPM2): if audio is unrealistically long for
    # the text — model went runaway, retry (different seed on retry if random).
    fixed_seed = seed is not None and int(seed) >= 0
    limit_sec = 0.13 * max(len(text), 1) + 3.0
    attempts = 1 if fixed_seed else 3
    audio = None
    for a in range(attempts):
        audio, was_cancelled = _generate_stream(m, _tok, text, label=label, attempt=(a + 1, attempts), **kw)
        if was_cancelled:
            return SR, np.zeros(0, np.float32)
        sec = (audio.shape[-1] if hasattr(audio, "shape") else len(audio)) / SR
        if sec <= limit_sec or a == attempts - 1:
            break
        print(f"[gen] runaway {sec:.1f}s > {limit_sec:.1f}s — retry {a + 2}/{attempts}", flush=True)
    return SR, audio.detach().cpu().numpy().astype(np.float32)


TARGET_LUFS = -16.0              # podcast/TTS standard (EBU R128, Google Assistant)
_PEAK_CEIL = 10 ** (-1.0 / 20)   # −1 dBFS — mix protection from clipping
_MAX_GAIN = 10 ** (20.0 / 20)    # do not boost quiet fragments by more than +20 dB (noise/silence)


def _loudness_normalize(x, sr=SR):
    """Fragment -> target loudness, so speakers in mix sound balanced (one not quieter than another).
    BS.1770 LUFS meter (pyloudnorm) if available, otherwise RMS fallback using pure numpy."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(sr).integrated_loudness(x)
        if np.isfinite(loud):
            gain = 10 ** ((TARGET_LUFS - loud) / 20)
            return (x * min(gain, _MAX_GAIN)).astype(np.float32)
    except Exception:
        pass
    rms = float(np.sqrt(np.mean(x ** 2)))   # fallback: RMS to ~−20 dBFS (speech reference)
    if rms < 1e-6:
        return x
    return (x * min((10 ** (-20.0 / 20)) / rms, _MAX_GAIN)).astype(np.float32)


def _peak_limit(x, ceil=_PEAK_CEIL):
    import numpy as np
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    return (x * (ceil / peak)).astype(np.float32) if peak > ceil else x


def _concat(chunks, gap=0.3, normalize=True):
    """Concatenate fragments with silence pause. normalize=True balances speaker loudness
    (LUFS/RMS per fragment) and applies −1 dBFS peak limit on final mix."""
    import numpy as np
    chunks = [c for c in chunks if c is not None and len(c)]
    if not chunks:
        return np.zeros(0, np.float32)
    if normalize:
        chunks = [_loudness_normalize(c) for c in chunks]
    sil = np.zeros(int(SR * gap), np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(sil)
        out.append(c)
    mix = np.concatenate(out)
    return _peak_limit(mix) if normalize else mix


def synth_longform(paragraphs, ref_audio=None, ref_text=None, **kw):
    """Long text by paragraphs. First chunk sets voice, its audio serves as reference for the rest."""
    import tempfile
    import soundfile as sf
    chunks = []
    chain_ref, chain_txt = ref_audio, ref_text
    paras = [p for p in paragraphs if p and p.strip()]
    for i, para in enumerate(paras):
        if _CANCEL:
            print(f"[gen] STOP halted at chunk {i + 1}/{len(paras)}", flush=True)
            break
        print(f"[gen] long-form: chunk {i + 1}/{len(paras)}", flush=True)
        _, a = generate(para, ref_audio=chain_ref, ref_text=chain_txt, **kw)
        chunks.append(a)
        if i == 0 and not _MOCK and ref_audio is None and len(a):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.close()  # Windows: close handle before re-opening in sf.write
            sf.write(f.name, a, SR)
            chain_ref, chain_txt = f.name, para
    return SR, _concat(chunks)


def synth_turns(turns, gap=0.4, **kw):
    """turns: [{'text','ref_audio','ref_text'}] — each speaker gets their own voice via their reference."""
    chunks = []
    for t in turns:
        if _CANCEL:
            break
        if t.get("text", "").strip():
            _, a = generate(t["text"], ref_audio=t.get("ref_audio"), ref_text=t.get("ref_text"), **kw)
            chunks.append(a)
    return SR, _concat(chunks, gap)
