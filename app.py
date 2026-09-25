"""Higgs Audio Studio — portable build by Nerual Dreming + Neuro-Soft.

Higgs Audio v3 TTS (100+ languages, cloning) + AI text director (meaning-based tags) +
multi-speaker Podcast and Audiobook modes. Bilingual RU/EN UI, dark theme.
"""
import os
import re
import sys
import asyncio

# Project root in sys.path — embedded python does not add script directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows retry_open patch for anyio/aiofiles (PermissionError from antivirus)
if sys.platform == "win32":
    try:
        import anyio
        import anyio._core._fileio
        _orig_open = anyio._core._fileio.open_file

        async def _retry_open(file, *a, **k):
            delay = 0.2
            for i in range(20):
                try:
                    return await _orig_open(file, *a, **k)
                except PermissionError:
                    if i == 19:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 1.2

        anyio._core._fileio.open_file = _retry_open
        anyio.open_file = _retry_open
    except Exception:
        pass

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

from datetime import datetime
from pathlib import Path

import gradio as gr

import higgs_engine as eng
import director as dr

SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "output"
VOICES_DIR = SCRIPT_DIR / "voices"
OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)

APP_NAME = "Higgs Audio Studio"
DEVICE_INFO = eng.device_info()
MODEL_CHOICES = list(dr.MODELS.keys())
MAX_SPK = 4
OWN_FILE = "— upload file / own file —"

CLOUD_VOICES_REPO = "Slait/russia_voices"


# ----------------------------------------------------------------------------
# Branding & Headers
# ----------------------------------------------------------------------------
_FLAG = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg"

_DONATE_RU = """
<div class="donate-popover">
  <p class="donate-intro">Привет! Я Илья (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), я создаю AI-инструменты, которые работают локально — бесплатно, без облака, без подписок. Ваш донат позволяет фокусироваться на исследовании и создании новых открытых проектов, а не на выживании. Спасибо!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Карта / PayPal (рубли, доллары, евро)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Ежемесячная подписка на Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""
_DONATE_EN = """
<div class="donate-popover">
  <p class="donate-intro">Hi! I'm Ilya (<a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a>), I build AI tools that run locally — for free, without cloud, without subscriptions. Your donation lets me focus on research and new open-source projects instead of just surviving. Thank you!</p>
  <div class="donate-sep"></div>
  <div class="donate-row"><a href="https://dalink.to/nerual_dreming" target="_blank">💳 Card / PayPal (USD, EUR, RUB)</a></div>
  <div class="donate-row"><a href="https://boosty.to/neuro_art" target="_blank">🚀 Monthly subscription on Boosty</a></div>
  <div class="donate-sep"></div>
  <div class="donate-row"><span>BTC</span><code>1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC</code></div>
  <div class="donate-row"><span>ETH</span><code>0xb5db65adf478983186d4897ba92fe2c25c594a0c</code></div>
  <div class="donate-row"><span>USDT TRC20</span><code>TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C</code></div>
</div>
"""


def _brand(subtitle, credits, donate, donate_label):
    return f"""
<div class="brand-header">
  <div class="lang-switcher">
    <a href="?__lang=en&amp;__theme=dark" class="lang-btn"><img src="{_FLAG}/1f1ec-1f1e7.svg" width="16" height="16"/>EN</a>
    <a href="?__lang=ru&amp;__theme=dark" class="lang-btn"><img src="{_FLAG}/1f1f7-1f1fa.svg" width="16" height="16"/>RU</a>
    <details class="donate-wrap"><summary class="lang-btn donate-btn"><img src="{_FLAG}/1fa99.svg" width="16" height="16"/>{donate_label}</summary>{donate}</details>
  </div>
  <div class="brand-box">
    <div class="brand-title">🎙️ {APP_NAME}</div>
    <div class="brand-subtitle">{subtitle}</div>
    <div class="brand-credits">{credits}</div>
    <div class="device-badge">💻 {DEVICE_INFO}</div>
  </div>
</div>
"""


BRAND_HTML_RU = _brand(
    "Higgs Audio v3 · 100+ языков · клонирование · AI-режиссёр текста · подкаст и аудиокнига",
    'Собрал <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'основатель <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'техноблогер и нейро-евангелист. Канал '
    '<a href="https://t.me/neuroport" target="_blank">Нейро-Софт</a> — репаки и портативки нейросетей.',
    _DONATE_RU, "Донат",
)
BRAND_HTML_EN = _brand(
    "Higgs Audio v3 · 100+ languages · voice cloning · AI text director · podcast & audiobook",
    'Built by <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> — '
    'founder of <a href="https://artgeneration.me" target="_blank">ArtGeneration.me</a>, '
    'tech-blogger and neuro-evangelist. Channel '
    '<a href="https://t.me/neuroport" target="_blank">Neuro-Soft</a> — portable AI builds.',
    _DONATE_EN, "Donate",
)


def _legend(emo, pro, sty, sfx, fmt):
    w = dr.WHITELIST
    return (f"**{emo}** " + ", ".join(sorted(w["emotion"])) + "\n\n"
            f"**{pro}** " + ", ".join(sorted(w["prosody"])) + "\n\n"
            f"**{sty}** " + ", ".join(sorted(w["style"])) + "\n\n"
            f"**{sfx}** " + ", ".join(sorted(w["sfx"])) + "\n\n" + fmt)


_LEGEND_RU = _legend("Эмоции (в начале предложения):", "Просодия:", "Стиль:",
                     "Звуки (внутри строки, тег вплотную):",
                     "Формат: `<|category:value|>`. Пример: `<|emotion:elation|>Привет! <|sfx:laughter|>ха-ха`")
_LEGEND_EN = _legend("Emotions (at sentence start):", "Prosody:", "Style:", "Sounds (inline, tag attached):",
                     "Format: `<|category:value|>`. Example: `<|emotion:elation|>Hi! <|sfx:laughter|>ha-ha`")

_PODFMT_RU = "**Формат сценария:** каждая строка `Speaker 0: реплика` / `Speaker 1: …` (также `Диктор N:`, `[N]`). Номер = диктор ниже."
_PODFMT_EN = "**Script format:** one line per turn `Speaker 0: line` / `Speaker 1: …` (also `[N]`). The number = speaker below."
_BOOKFMT_RU = "**Формат:** `Speaker 0:` — рассказчик, `Speaker 1+:` — персонажи. Разметь кнопкой или вручную, потом озвучь."
_BOOKFMT_EN = "**Format:** `Speaker 0:` — narrator, `Speaker 1+:` — characters. Attribute with the button or by hand, then synthesize."

TAG_DESC_EN = {
    "affection": "Warmth, tenderness", "amusement": "Amusement, playful chuckle", "anger": "Anger",
    "arousal": "Heightened arousal", "awe": "Awe, admiration", "bitterness": "Bitterness",
    "confusion": "Confusion, perplexity", "contemplation": "Contemplation, reflection", "contentment": "Quiet contentment",
    "determination": "Determination, firmness", "disgust": "Disgust", "elation": "Elation, joy",
    "enthusiasm": "Enthusiasm, excitement", "fear": "Fear", "helplessness": "Helplessness",
    "longing": "Longing, yearning", "pride": "Pride, confidence", "relief": "Relief",
    "sadness": "Sadness", "shame": "Shame", "surprise": "Surprise",
    "speed_very_slow": "Very slow (≈0.65×)", "speed_slow": "Slow (≈0.85×)",
    "speed_fast": "Fast (≈1.2×)", "speed_very_fast": "Very fast (≈1.4×)",
    "pitch_low": "Lower pitch (≈−3 semitones)", "pitch_high": "Higher pitch (≈+2.5 semitones)",
    "expressive_high": "More expressive", "expressive_low": "Flatter, monotone",
    "pause": "Pause ≈400–700 ms", "long_pause": "Long pause ≈700–1500 ms",
    "singing": "Singing", "shouting": "Shouting, projection", "whispering": "Whispering",
    "cough": "Cough (sound: ahem)", "laughter": "Laughter (ha-ha)", "crying": "Crying (sob)",
    "screaming": "Screaming (aaah)", "burping": "Burping", "humming": "Humming (mmm)",
    "sigh": "Sigh", "sniff": "Sniffing", "sneeze": "Sneeze (achoo)",
}
TAG_DESC_RU = {
    "affection": "Теплота, нежность", "amusement": "Веселье, игривый смешок", "anger": "Гнев",
    "arousal": "Обострённое желание", "awe": "Благоговение, восхищение", "bitterness": "Горечь",
    "confusion": "Растерянность", "contemplation": "Задумчивость, рефлексия", "contentment": "Спокойное удовлетворение",
    "determination": "Решимость, твёрдость", "disgust": "Отвращение", "elation": "Ликование, радость",
    "enthusiasm": "Энтузиазм, воодушевление", "fear": "Страх", "helplessness": "Беспомощность",
    "longing": "Тоска, томление", "pride": "Гордость, уверенность", "relief": "Облегчение",
    "sadness": "Грусть", "shame": "Стыд", "surprise": "Удивление",
    "speed_very_slow": "Очень медленно (≈0.65×)", "speed_slow": "Медленно (≈0.85×)",
    "speed_fast": "Быстро (≈1.2×)", "speed_very_fast": "Очень быстро (≈1.4×)",
    "pitch_low": "Ниже тон (≈−3 полутона)", "pitch_high": "Выше тон (≈+2.5 полутона)",
    "expressive_high": "Выразительнее", "expressive_low": "Ровнее, монотоннее",
    "pause": "Пауза ≈400–700 мс (по месту)", "long_pause": "Длинная пауза ≈700–1500 мс (по месту)",
    "singing": "Пение", "shouting": "Крик, посыл голоса", "whispering": "Шёпот",
    "cough": "Кашель (звук: кхм)", "laughter": "Смех (ха-ха)", "crying": "Плач (ыыы)",
    "screaming": "Крик (ааа)", "burping": "Отрыжка", "humming": "Мычание (ммм)",
    "sigh": "Вздох (эх)", "sniff": "Шмыганье носом", "sneeze": "Чихание (апчхи)",
}

_CAT_NAMES_EN = {"emotion": "😊 Emotions (sentence start)", "prosody": "🎵 Prosody",
                 "style": "🎭 Style (sentence start)", "sfx": "🔊 Sounds (inline, attached)"}
_CAT_NAMES_RU = {"emotion": "😊 Эмоции (в начало предложения)", "prosody": "🎵 Просодия",
                 "style": "🎭 Стиль (в начало)", "sfx": "🔊 Звуки (по месту, рядом со звукоподражанием)"}

TAGS_LEGEND_MD_EN = "\n\n".join(
    f"**{_CAT_NAMES_EN[c]}**\n" + "\n".join(f"- `<|{c}:{v}|>` — {TAG_DESC_EN.get(v, '')}" for v in sorted(dr.WHITELIST[c]))
    for c in ("emotion", "prosody", "style", "sfx")
)
TAGS_LEGEND_MD_RU = "\n\n".join(
    f"**{_CAT_NAMES_RU[c]}**\n" + "\n".join(f"- `<|{c}:{v}|>` — {TAG_DESC_RU.get(v, '')}" for v in sorted(dr.WHITELIST[c]))
    for c in ("emotion", "prosody", "style", "sfx")
)

# ----------------------------------------------------------------------------
# i18n (gr.I18n, registered in launch(i18n=))
# ----------------------------------------------------------------------------
_RU = {
    "tab_tts": "🎙️ Озвучка", "tab_expr": "🎭 Экспрессия + Режиссёр", "tab_clone": "🧬 Клонирование",
    "tab_pod": "🎬 Подкаст", "tab_book": "📚 Аудиокнига", "tab_batch": "📦 Пакет",
    "text": "Текст", "ph_text": "Введите текст… можно вставлять теги: <|emotion:elation|>Привет!",
    "generate": "🔊 Озвучить", "stop": "⏹ Стоп", "result": "Результат", "advanced": "Доп. настройки",
    "director_model": "Модель режиссёра (для обогащения / диалогов)",
    "quant": "Квантизация Higgs ⚗️ (экспериментально)",
    "quant_info": "⚗️ Экспериментально: квантизация (4/8-bit) экономит VRAM, но качество звука заметно страдает. Для лучшего качества оставьте bf16.",
    "out_format": "Формат вывода",
    "cat_emotion": "😊 Эмоции (на предложение)", "cat_prosody": "🎵 Просодия", "cat_style": "🎭 Стиль", "cat_sfx": "🔊 Звуки (по месту в тексте)",
    "download_all": "⬇️ Скачать все 700+",
    "enrich": "✨ Обогатить текст", "auto_enrich": "✨ Авто-обогащение промпта режиссёром",
    "ref_voice": "Аудио-референс (голос)", "ref_text": "Транскрипт референса (заполнится сам)",
    "ph_clone_tr": "Что произносится в референсе…", "voice_preset": "Пресет голоса",
    "refresh": "🔄 Обновить", "transcribe_btn": "📝 Распознать транскрипт",
    "seed": "Сид (-1 = случайно)", "max_tokens": "Макс. токенов",
    "examples": "Примеры", "tags_help": "❓ Все теги (подсказка)", "tags_legend": TAGS_LEGEND_MD_RU,
    "ph_clone": "Текст, который произнесёт клонированный голос…",
    "cloud_title": "☁️ Скачать голоса с сервера (русский пак)", "cloud_status": "Статус",
    "load_list": "Обновить список", "cloud_voices": "Доступные голоса", "download_sel": "⬇️ Скачать выбранные",
    "refresh_voices": "🔄 Обновить список голосов",
    "num_speakers": "Количество дикторов", "pod_hint": "Опиши тему — режиссёр напишет диалог. Затем задай голоса дикторам и нажми «Озвучить».",
    "pod_format": _PODFMT_RU, "topic": "Тема подкаста", "ph_topic": "Напр.: плюсы и минусы локального ИИ дома",
    "make_script": "📝 Сгенерировать сценарий", "script": "Сценарий (можно править)",
    "ph_script": "Speaker 0: Привет!\nSpeaker 1: Здравствуй!", "synth": "🔊 Озвучить",
    "book_hint": "Вставь текст книги/главы, задай голоса (Speaker 0 — рассказчик), размечай по ролям и озвучивай.",
    "book_format": _BOOKFMT_RU, "book_text": "Текст книги / главы",
    "ph_book": "Вставь фрагмент с репликами персонажей…", "markup": "📝 Разметить по ролям",
    "batch_text": "Список текстов (по одному в строке)", "ph_batch": "Первая фраза.\nВторая фраза.\nТретья фраза.",
    "log": "Лог", "brand_header_html": BRAND_HTML_RU,
}
_EN = {
    "tab_tts": "🎙️ TTS", "tab_expr": "🎭 Expressive + Director", "tab_clone": "🧬 Cloning",
    "tab_pod": "🎬 Podcast", "tab_book": "📚 Audiobook", "tab_batch": "📦 Batch",
    "text": "Text", "ph_text": "Type text… you can insert tags: <|emotion:elation|>Hi!",
    "generate": "🔊 Generate", "stop": "⏹ Stop", "result": "Result", "advanced": "Advanced",
    "director_model": "Director model (enrich / dialogues)",
    "quant": "Higgs quantization ⚗️ (experimental)",
    "quant_info": "⚗️ Experimental: quantization (4/8-bit) saves VRAM but audibly degrades quality. Keep bf16 for best quality.",
    "out_format": "Output format",
    "cat_emotion": "😊 Emotion (per sentence)", "cat_prosody": "🎵 Prosody", "cat_style": "🎭 Style", "cat_sfx": "🔊 Sounds (inline)",
    "download_all": "⬇️ Download all presets",
    "enrich": "✨ Enrich text", "auto_enrich": "✨ Auto-enrich prompt with director",
    "ref_voice": "Reference audio (voice)", "ref_text": "Reference transcript (auto-filled)",
    "ph_clone_tr": "What the reference says…", "voice_preset": "Voice preset",
    "refresh": "🔄 Refresh", "transcribe_btn": "📝 Transcribe reference",
    "seed": "Seed (-1 = random)", "max_tokens": "Max tokens",
    "examples": "Examples", "tags_help": "❓ All tags (legend)", "tags_legend": TAGS_LEGEND_MD_EN,
    "ph_clone": "Text the cloned voice will speak…",
    "cloud_title": "☁️ Download cloud voice presets", "cloud_status": "Status",
    "load_list": "Refresh list", "cloud_voices": "Available voices", "download_sel": "⬇️ Download selected",
    "refresh_voices": "🔄 Refresh voice list",
    "num_speakers": "Number of speakers", "pod_hint": "Describe a topic — the director writes a dialogue. Then set speaker voices and synthesize.",
    "pod_format": _PODFMT_EN, "topic": "Podcast topic", "ph_topic": "e.g. pros and cons of local AI at home",
    "make_script": "📝 Generate script", "script": "Script (editable)",
    "ph_script": "Speaker 0: Hello!\nSpeaker 1: Hi there!", "synth": "🔊 Synthesize",
    "book_hint": "Paste book/chapter text, set voices (Speaker 0 = narrator), attribute roles and synthesize.",
    "book_format": _BOOKFMT_EN, "book_text": "Book / chapter text",
    "ph_book": "Paste a passage with character dialogue…", "markup": "📝 Attribute roles",
    "batch_text": "List of texts (one per line)", "ph_batch": "First line.\nSecond line.\nThird line.",
    "log": "Log", "brand_header_html": BRAND_HTML_EN,
}
I18N = gr.I18n(en=_EN, ru=_RU)


def T(key):
    return I18N(key)


HEAD_SCRIPT = """
<script>
(function(){
  var lang;
  try { lang = new URL(window.location).searchParams.get('__lang'); } catch(e) { lang = null; }
  if (!lang) return;
  try {
    Object.defineProperty(navigator, 'language',  {get: function(){ return lang; }, configurable: true});
    Object.defineProperty(navigator, 'languages', {get: function(){ return [lang]; }, configurable: true});
    document.documentElement.lang = lang;
  } catch(e) {}
  var sp = null;
  function getStore(){
    if (sp) return sp;
    var link = document.querySelector('link[href*="i18n-"]');
    if (!link) return null;
    sp = import(link.href).then(function(m){
      for (var k in m){ var v = m[k];
        try { if (v && typeof v.subscribe === 'function' && typeof v.set === 'function'){
          var c; var u = v.subscribe(function(x){ c = x; }); if (typeof u === 'function') u();
          if (typeof c === 'string' && /^[a-z]{2}(-[A-Za-z]+)?$/.test(c)) return v;
        }} catch(e) {}
      }
      return null;
    }).catch(function(){ return null; });
    return sp;
  }
  function apply(){
    var p = getStore(); if (!p) return;
    p.then(function(s){ if (s){ try { s.set(lang); window.dispatchEvent(new Event('languagechange')); } catch(e) {} } });
  }
  var n = 0, iv = setInterval(function(){ apply(); if (++n > 30) clearInterval(iv); }, 120);
  apply();
  document.addEventListener('click', function(e){
    try { if (e.target && e.target.closest && e.target.closest('[role=tablist],.tab-nav,[role=tab]')){
      setTimeout(apply, 60); setTimeout(apply, 250);
    }} catch(_) {}
  }, true);
})();
</script>
"""

CSS = """
.gradio-container, .gradio-container * {
  font-family: Arial, Helvetica, sans-serif !important;
}
.gradio-container code, .gradio-container pre, .gradio-container kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
}
.gradio-container {max-width: 1080px !important; margin: auto !important;}
.brand-header { position: relative; }
.brand-box { background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #7e22ce 100%);
  padding: 24px 28px; border-radius: 16px; margin: 8px 0 16px 0;
  box-shadow: 0 10px 30px rgba(109,40,217,0.35); color: white; text-align: center; }
.brand-title { font-size: 1.9em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { font-size: 1em; opacity: 0.9; margin-bottom: 14px; }
.brand-credits { font-size: 0.9em; opacity: 0.95; }
.brand-credits a { color:#fbbf24 !important; text-decoration:none !important; font-weight:600 !important; background:none !important; padding:0 !important; }
.brand-credits a:hover { text-decoration: underline !important; }
.device-badge { display:inline-block; background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:999px; font-size:0.85em; margin-top:10px; }
.lang-switcher { position:absolute; top:12px; right:16px; display:flex; gap:6px; align-items:flex-start; z-index:50; }
.lang-btn { background:rgba(255,255,255,0.18); color:white !important; padding:5px 10px; border-radius:8px;
  font-size:0.82em; text-decoration:none !important; font-weight:600; display:inline-flex; flex-direction:column;
  align-items:center; justify-content:center; gap:2px; line-height:1; white-space:nowrap; min-width:44px; cursor:pointer; }
.lang-btn:hover { background:rgba(255,255,255,0.3); }
.lang-btn img { margin:0 !important; vertical-align:middle !important; }
.tabs > div[role="tablist"] > button, .tab-nav > button { flex:1 !important; text-align:center !important; }
.spk-block { background: rgba(124,58,237,0.06); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:10px; margin:6px 0; }
.donate-wrap { position:relative; display:inline-block; }
.donate-wrap > summary.donate-btn { list-style:none; cursor:pointer; }
.donate-wrap > summary.donate-btn::-webkit-details-marker { display:none; }
.donate-wrap:not([open]) .donate-popover { display:none !important; pointer-events:none; }
.donate-popover { position:absolute; top:calc(100% + 6px); right:0; background:rgba(20,20,28,0.98);
  backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.12); border-radius:10px; padding:10px 14px;
  min-width:320px; box-shadow:0 8px 24px rgba(0,0,0,0.4); z-index:999; font-size:13px; line-height:1.5; text-align:left; }
.donate-popover a { color:#c4a3ff !important; text-decoration:none !important; font-weight:600 !important; background:none !important; padding:0 !important; }
.donate-row { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:4px 0; }
.donate-row > span { color:#9ca3af; font-weight:600; font-size:12px; flex:0 0 80px; text-align:left; white-space:nowrap; }
.donate-row > code { flex:1; text-align:left; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px; font-size:11px; color:#e5e7eb; user-select:all; }
.donate-intro { color:#cbd5e1; font-size:12px; line-height:1.5; margin:0 0 4px 0; }
.donate-sep { height:1px; background:rgba(255,255,255,0.1); margin:6px 0; }
.gradio-container { --block-border-color: rgba(255,255,255,0.10) !important;
  --border-color-primary: rgba(255,255,255,0.10) !important;
  --input-border-color: rgba(255,255,255,0.10) !important;
  --neutral-200: rgba(255,255,255,0.10) !important; }
.gradio-container .block { border-color: rgba(255,255,255,0.10) !important; }
.gradio-container input[type=range] { accent-color: #7c3aed; }
.tagbtn { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.tagbtn button { white-space: pre-line !important; line-height: 1.15 !important; height: auto !important; min-height: 0 !important; text-align: center !important; padding: 5px 11px !important; font-size: 0.8em !important; }
.tagbtn button > * { display: block; }
"""

import json as _json

DARK_JS = ("""
() => {
  try { const u=new URL(window.location);
    if(u.searchParams.get('__theme')!=='dark' && !sessionStorage.getItem('_hgs_dark')){
      sessionStorage.setItem('_hgs_dark','1'); u.searchParams.set('__theme','dark'); window.location.replace(u.href); return;
    } } catch(e){}
  const TD_EN = __TAGDESC_EN__;
  const TD_RU = __TAGDESC_RU__;
  const apply = () => {
    let isRu = false;
    try { isRu = new URL(window.location).searchParams.get('__lang') === 'ru'; } catch(e){}
    const TD = isRu ? TD_RU : TD_EN;
    document.querySelectorAll('.tagbtn button').forEach(b => {
      const k = (b.textContent||'').trim().split(/[\\s\\n]+/)[0];
      if (k && TD[k]) b.title = TD[k];
    });
  };
  apply(); setInterval(apply, 1200);
}
""").replace("__TAGDESC_EN__", _json.dumps(TAG_DESC_EN, ensure_ascii=False)).replace("__TAGDESC_RU__", _json.dumps(TAG_DESC_RU, ensure_ascii=False))

TTS_EXAMPLES = [
    ["Hello! This is Higgs Audio Studio — local speech synthesis in over 100 languages."],
    ["<|emotion:elation|>Incredible, we did it! <|sfx:laughter|>ha-ha-ha!"],
    ["<|style:whispering|>Come closer, I will tell you a secret."],
    ["This model delivers high-fidelity expressive voice generation fully offline."],
]
EXPR_EXAMPLES = [
    ["<|emotion:sadness|>I am so sorry that it happened this way. <|prosody:long_pause|> But we will get through this."],
    ["<|emotion:anger|>How much longer do we have to endure this?! <|sfx:sigh|>sigh…"],
    ["<|style:shouting|>Keep pushing! The finish line is right ahead!"],
]
POD_TOPICS = [["Pros and cons of local AI at home"], ["How AI is changing music production"],
              ["The future of voice assistants"]]
BOOK_EXAMPLES = [["The old lighthouse had been silent for years. 'Is anybody here?' Tom called out as he climbed the creaking spiral stairs. Only silence answered."]]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
_OUT_FORMAT = "mp3"
_FMT = {"wav": ("WAV", None), "mp3": ("MP3", None), "flac": ("FLAC", None), "ogg": ("OGG", "VORBIS")}


def set_out_format(f):
    global _OUT_FORMAT
    _OUT_FORMAT = f if f in _FMT else "wav"


def _has_audio(wav):
    try:
        return wav is not None and len(wav) > 0
    except TypeError:
        return False


def _safe_audio_output(sr, wav, ctx="tts"):
    if not _has_audio(wav):
        print(f"[{ctx}] empty audio result; returning None to Gradio to avoid crash", flush=True)
        return None
    import numpy as np
    wav_int16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    return sr, wav_int16


def _save(sr, wav, prefix="tts"):
    import soundfile as sf
    if not _has_audio(wav):
        return None
    fmt = _OUT_FORMAT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container, subtype = _FMT.get(fmt, ("WAV", None))
    path = OUTPUT_DIR / f"{prefix}_{stamp}.{fmt}"
    try:
        sf.write(str(path), wav, sr, format=container, subtype=subtype)
    except Exception as e:
        print(f"[save] format {fmt} could not be written ({e}) → wav")
        path = OUTPUT_DIR / f"{prefix}_{stamp}.wav"
        sf.write(str(path), wav, sr)
    return str(path)


def scan_voices():
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return sorted(p.stem for p in VOICES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts)


def voice_path(name):
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return str(p)
    return None


def voice_transcript(name):
    for ext in (".txt", ".lab"):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            for enc in ("utf-8", "cp1251"):
                try:
                    return p.read_text(encoding=enc).strip()
                except Exception:
                    continue
    return ""


def cb_preset(name):
    """Preset -> populate audio + transcript."""
    if not name or name == OWN_FILE or name.startswith("—"):
        return None, ""
    return voice_path(name), voice_transcript(name)


_ASR = None


def _get_asr():
    global _ASR
    if _ASR is None:
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        proc = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
        amodel = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base").eval()
        _ASR = (proc, amodel)
    return _ASR


def transcribe(ref_audio):
    if not ref_audio:
        return gr.update()
    if eng._MOCK:
        return "sample transcript (mock)"
    try:
        import torch
        import soundfile as sf
        import torchaudio
        proc, amodel = _get_asr()
        data, sr = sf.read(ref_audio, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        inp = proc(wav.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            tok = amodel.generate(**inp)
        return proc.decode(tok[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[asr] {e}")
        return gr.update()


_SPK_PATTERNS = [r'^speaker\s*(\d+)\s*:\s*(.+)$', r'^диктор\s*(\d+)\s*:\s*(.+)$',
                 r'^голос\s*(\d+)\s*:\s*(.+)$', r'^\[(\d+)\]\s*(.+)$']


def parse_script(script):
    """'Speaker N: text' (or [N]) → [(speaker_id, text)]; unrecognized lines → Speaker 0."""
    out = []
    for line in (script or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        matched = False
        for pat in _SPK_PATTERNS:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                out.append((int(m.group(1)), m.group(2).strip()))
                matched = True
                break
        if not matched:
            out.append((0, line))
    return out


def _chunk(text, max_chars=120):
    """Chunk long text (paragraphs, then sentences) for long-form synthesis."""
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        cur = ""
        for s in re.split(r"(?<=[.!?…])\s+", para):
            if cur and len(cur) + len(s) > max_chars:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
    return chunks


def _speak(text, ref_audio=None, ref_text=None, **kw):
    """Short text -> single pass; long text -> long-form with timbre transfer."""
    chunks = _chunk(text)
    if len(chunks) <= 1:
        return eng.generate(text, ref_audio=ref_audio, ref_text=ref_text, **kw)
    return eng.synth_longform(chunks, ref_audio=ref_audio, ref_text=ref_text, **kw)


def cb_load_cloud():
    """List cloud voice presets (Slait/russia_voices dataset)."""
    voices = []
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = sorted(f[:-4] for f in files if f.endswith(".mp3"))
    except Exception as e:
        print(f"[voices] list: {e}")
    status = f"Found: {len(voices)}" if voices else "Failed to load"
    return status, gr.update(choices=voices, value=[])


def _dl_voice(name):
    """Download a single cloud voice preset via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(CLOUD_VOICES_REPO, f"{name}.mp3", repo_type="dataset", local_dir=str(VOICES_DIR))
        try:
            hf_hub_download(CLOUD_VOICES_REPO, f"{name}.txt", repo_type="dataset", local_dir=str(VOICES_DIR))
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[voices] dl {name}: {e}")
        return False


def cb_download_voices(selected):
    if not selected:
        return "Select voices", gr.update()
    ok = sum(_dl_voice(n) for n in selected)
    return f"Downloaded: {ok}/{len(selected)}", gr.update(choices=[OWN_FILE] + scan_voices())


def cb_download_all_cloud(progress=gr.Progress()):
    """Download the full cloud voice library."""
    try:
        from huggingface_hub import list_repo_files
        names = sorted(f[:-4] for f in list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset") if f.endswith(".mp3"))
    except Exception as e:
        return f"List error: {e}", gr.update()
    if not names:
        return "Empty list", gr.update()
    ok = 0
    for i, name in enumerate(names):
        progress((i + 1) / len(names), desc=f"{i + 1}/{len(names)} · {name}")
        if _dl_voice(name):
            ok += 1
    return f"Downloaded: {ok}/{len(names)}", gr.update(choices=[OWN_FILE] + scan_voices())


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def _maybe_enrich(text, model, auto):
    return dr.enrich(text, model) if auto else text


def cb_tts(text, model, auto, temperature, top_p, top_k, max_new, seed):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    sr, wav = _speak(text, temperature=temperature, top_p=top_p, top_k=top_k,
                     max_new_tokens=max_new, seed=seed)
    audio = _safe_audio_output(sr, wav, ctx="tts")
    if audio is None:
        return None, text
    _save(sr, wav, "tts")
    return audio, text


def cb_enrich(text, model):
    return dr.enrich(text, label=model)


def cb_expr(text, model, auto):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    sr, wav = _speak(text)
    audio = _safe_audio_output(sr, wav, ctx="expr")
    if audio is None:
        return None, text
    _save(sr, wav, "expr")
    return audio, text


def cb_clone(text, model, auto, ref_audio, ref_text, preset, temperature, top_p, seed):
    eng.clear_cancel()
    text = _maybe_enrich(text, model, auto)
    ref = ref_audio or (voice_path(preset) if preset and preset != OWN_FILE and not preset.startswith("—") else None)
    sr, wav = _speak(text, ref_audio=ref, ref_text=ref_text, temperature=temperature, top_p=top_p, seed=seed)
    audio = _safe_audio_output(sr, wav, ctx="clone")
    if audio is None:
        return None, text
    _save(sr, wav, "clone")
    return audio, text


def cb_podcast_script(topic, num, model):
    return dr.write_podcast(topic, int(num), label=model)


def cb_book_markup(text, num, model):
    return dr.cast_audiobook(text, int(num), label=model)


def cb_multi_synth(script, a0, a1, a2, a3, t0, t1, t2, t3, progress=gr.Progress()):
    """Parse 'Speaker N:' → synthesize turns with speaker voice N → concat with normalization."""
    eng.clear_cancel()
    audios = [a0, a1, a2, a3]
    texts = [t0, t1, t2, t3]
    turns = [(sid, txt) for sid, txt in parse_script(script) if txt.strip()]
    if not turns:
        return None
    chunks = []
    for i, (sid, txt) in enumerate(turns):
        if eng.cancelled():
            break
        progress((i + 1) / len(turns), desc=f"{i + 1}/{len(turns)} · Speaker {sid}")
        ref = audios[sid] if 0 <= sid < MAX_SPK else None
        rt = (texts[sid] if 0 <= sid < MAX_SPK else None) or None
        _, wav = _speak(txt, ref_audio=ref, ref_text=rt)
        if wav is not None and len(wav):
            chunks.append(wav)
    final = eng._concat(chunks, gap=0.3)
    _save(eng.SR, final, "multi")
    return (eng.SR, final)


def cb_batch(texts, model, auto, progress=gr.Progress()):
    eng.clear_cancel()
    lines = [t.strip() for t in (texts or "").splitlines() if t.strip()]
    log, paths = [], []
    for i, line in enumerate(lines):
        if eng.cancelled():
            yield "\n".join(log) + "\n\n⏹ Stopped.", paths
            return
        progress((i + 1) / max(len(lines), 1), desc=f"{i + 1}/{len(lines)}")
        if auto:
            line = dr.enrich(line, model)
        sr, wav = _speak(line)
        p = _save(sr, wav, "batch")
        if p:
            paths.append(p)
        log.append(f"✓ {i + 1}. {line[:60]}")
        yield "\n".join(log), paths
    yield "\n".join(log) + "\n\nDone.", paths


# ----------------------------------------------------------------------------
# UI Layout
# ----------------------------------------------------------------------------
def _speaker_blocks():
    """4 speaker blocks (preset + audio + transcript), visibility by slider."""
    with gr.Row():
        num = gr.Slider(2, MAX_SPK, value=2, step=1, label=T("num_speakers"))
        refresh = gr.Button(T("refresh_voices"), size="sm", scale=0)
    choices = [OWN_FILE] + scan_voices()
    blocks, audios, texts, pres = [], [], [], []
    for i in range(MAX_SPK):
        with gr.Group(visible=(i < 2), elem_classes="spk-block") as bl:
            gr.Markdown(f"**Speaker {i}**")
            pre = gr.Dropdown(choices, value=OWN_FILE, label=T("voice_preset"))
            au = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
            tx = gr.Textbox(label=T("ref_text"), lines=1, placeholder=T("ph_clone_tr"))
            pre.change(cb_preset, [pre], [au, tx])
        blocks.append(bl)
        audios.append(au)
        texts.append(tx)
        pres.append(pre)
    num.change(lambda n: [gr.update(visible=(i < n)) for i in range(MAX_SPK)], [num], blocks)
    refresh.click(lambda: [gr.update(choices=[OWN_FILE] + scan_voices()) for _ in range(MAX_SPK)], None, pres)
    return num, audios, texts


def build():
    with gr.Blocks(title=APP_NAME) as demo:
        gr.HTML(T("brand_header_html"))
        model_dd = gr.Dropdown(MODEL_CHOICES, value=dr.DEFAULT_MODEL, label=T("director_model"))
        quant_dd = gr.Dropdown([("bf16 — max quality (default)", "bf16"),
                                ("4-bit (nf4) ⚗️ experimental — lower quality", "4bit"),
                                ("8-bit ⚗️ experimental — lower quality", "8bit")],
                               value="bf16", label=T("quant"), info=T("quant_info"))
        quant_dd.change(lambda p: eng.set_precision(p), [quant_dd], None)
        fmt_dd = gr.Radio(["mp3", "wav", "flac", "ogg"], value="mp3", label=T("out_format"))
        fmt_dd.change(set_out_format, [fmt_dd], None)

        with gr.Tabs():
            # 1. TTS
            with gr.Tab(T("tab_tts")):
                with gr.Row():
                    with gr.Column():
                        t_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=4)
                        with gr.Accordion(T("advanced"), open=False):
                            t_temp = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="Temperature")
                            t_top_p = gr.Slider(0.1, 1.0, 0.95, step=0.01, label="Top-p")
                            t_top_k = gr.Slider(0, 1026, 50, step=1, label="Top-k (0=off)")
                            t_max = gr.Slider(64, 4096, 2048, step=64, label=T("max_tokens"))
                            t_seed = gr.Number(-1, label=T("seed"), precision=0)
                        t_auto = gr.Checkbox(label=T("auto_enrich"), value=False)
                        t_btn = gr.Button(T("generate"), variant="primary", size="lg")
                        t_stop = gr.Button(T("stop"), variant="stop")
                    t_out = gr.Audio(label=T("result"), type="numpy", autoplay=True)
                gr.Examples(TTS_EXAMPLES, inputs=[t_text], label=T("examples"))
                ev_tts = t_btn.click(cb_tts, [t_text, model_dd, t_auto, t_temp, t_top_p, t_top_k, t_max, t_seed],
                                     [t_out, t_text])
                t_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_tts])

            # 2. Expressive + Director
            with gr.Tab(T("tab_expr")):
                e_text = gr.Textbox(label=T("text"), placeholder=T("ph_text"), lines=5)
                e_auto = gr.Checkbox(label=T("auto_enrich"), value=False)
                for cat, clabel in (("emotion", T("cat_emotion")), ("prosody", T("cat_prosody")),
                                    ("style", T("cat_style")), ("sfx", T("cat_sfx"))):
                    gr.Markdown(f"**{clabel}**")
                    with gr.Row():
                        for val in sorted(dr.WHITELIST[cat]):
                            gr.Button(f"{val}", size="sm", elem_classes=["tagbtn"]).click(
                                lambda t, c=cat, v=val: (t or "") + f"<|{c}:{v}|>", [e_text], [e_text])
                with gr.Row():
                    e_enrich = gr.Button(T("enrich"), variant="secondary")
                    e_btn = gr.Button(T("generate"), variant="primary")
                e_stop = gr.Button(T("stop"), variant="stop")
                e_out = gr.Audio(label=T("result"), type="numpy", autoplay=True)
                gr.Examples(EXPR_EXAMPLES, inputs=[e_text], label=T("examples"))
                with gr.Accordion(T("tags_help"), open=True):
                    gr.Markdown(T("tags_legend"))
                e_enrich.click(cb_enrich, [e_text, model_dd], [e_text])
                ev_expr = e_btn.click(cb_expr, [e_text, model_dd, e_auto], [e_out, e_text])
                e_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_expr])

            # 3. Voice Cloning
            with gr.Tab(T("tab_clone")):
                with gr.Row():
                    with gr.Column():
                        c_text = gr.Textbox(label=T("text"), placeholder=T("ph_clone"), lines=3)
                        c_preset = gr.Dropdown([OWN_FILE] + scan_voices(), value=OWN_FILE, label=T("voice_preset"))
                        c_refresh = gr.Button(T("refresh"), size="sm")
                        c_ref = gr.Audio(label=T("ref_voice"), type="filepath", sources=["upload", "microphone"])
                        c_ref_text = gr.Textbox(label=T("ref_text"), lines=2, placeholder=T("ph_clone_tr"))
                        c_tr_btn = gr.Button(T("transcribe_btn"), size="sm")
                        c_temp = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="Temperature")
                        c_top_p = gr.Slider(0.1, 1.0, 0.95, step=0.01, label="Top-p")
                        c_seed = gr.Number(-1, label=T("seed"), precision=0)
                        c_auto = gr.Checkbox(label=T("auto_enrich"), value=False)
                        c_btn = gr.Button(T("generate"), variant="primary", size="lg")
                        c_stop = gr.Button(T("stop"), variant="stop")
                    c_out = gr.Audio(label=T("result"), type="numpy", autoplay=True)
                with gr.Accordion(T("cloud_title"), open=False):
                    cl_status = gr.Textbox(label=T("cloud_status"), interactive=False)
                    with gr.Row():
                        cl_load = gr.Button(T("load_list"), size="sm")
                        cl_all = gr.Button(T("download_all"), size="sm")
                    cl_voices = gr.CheckboxGroup(choices=[], label=T("cloud_voices"))
                    cl_dl = gr.Button(T("download_sel"), variant="primary", size="sm")
                c_preset.change(cb_preset, [c_preset], [c_ref, c_ref_text])
                c_tr_btn.click(transcribe, [c_ref], [c_ref_text])
                c_refresh.click(lambda: gr.update(choices=[OWN_FILE] + scan_voices()), None, [c_preset])
                cl_load.click(cb_load_cloud, None, [cl_status, cl_voices])
                cl_all.click(cb_download_all_cloud, None, [cl_status, c_preset])
                cl_dl.click(cb_download_voices, [cl_voices], [cl_status, c_preset])
                ev_clone = c_btn.click(cb_clone, [c_text, model_dd, c_auto, c_ref, c_ref_text, c_preset, c_temp, c_top_p, c_seed],
                                       [c_out, c_text])
                c_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_clone])

            # 4. Podcast
            with gr.Tab(T("tab_pod")):
                gr.Markdown(T("pod_hint"))
                gr.Markdown(T("pod_format"))
                p_num, p_audios, p_texts = _speaker_blocks()
                p_topic = gr.Textbox(label=T("topic"), placeholder=T("ph_topic"), lines=2)
                gr.Examples(POD_TOPICS, inputs=[p_topic], label=T("examples"))
                p_model = gr.Dropdown(MODEL_CHOICES, value=dr.DEFAULT_MODEL, label=T("director_model"))
                p_script_btn = gr.Button(T("make_script"), variant="secondary")
                p_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
                p_btn = gr.Button(T("synth"), variant="primary", size="lg")
                p_stop = gr.Button(T("stop"), variant="stop")
                p_out = gr.Audio(label=T("result"), type="numpy", autoplay=True)
                p_script_btn.click(cb_podcast_script, [p_topic, p_num, p_model], [p_script])
                ev_pod = p_btn.click(cb_multi_synth, [p_script] + p_audios + p_texts, p_out)
                p_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_pod])

            # 5. Audiobook
            with gr.Tab(T("tab_book")):
                gr.Markdown(T("book_hint"))
                gr.Markdown(T("book_format"))
                b_num, b_audios, b_texts = _speaker_blocks()
                b_text = gr.Textbox(label=T("book_text"), placeholder=T("ph_book"), lines=6)
                gr.Examples(BOOK_EXAMPLES, inputs=[b_text], label=T("examples"))
                b_model = gr.Dropdown(MODEL_CHOICES, value=dr.DEFAULT_MODEL, label=T("director_model"))
                b_markup = gr.Button(T("markup"), variant="secondary")
                b_script = gr.Textbox(label=T("script"), placeholder=T("ph_script"), lines=9)
                b_btn = gr.Button(T("synth"), variant="primary", size="lg")
                b_stop = gr.Button(T("stop"), variant="stop")
                b_out = gr.Audio(label=T("result"), type="numpy", autoplay=True)
                b_markup.click(cb_book_markup, [b_text, b_num, b_model], [b_script])
                ev_book = b_btn.click(cb_multi_synth, [b_script] + b_audios + b_texts, b_out)
                b_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_book])

            # 6. Batch
            with gr.Tab(T("tab_batch")):
                bt_text = gr.Textbox(label=T("batch_text"), placeholder=T("ph_batch"), lines=6)
                bt_auto = gr.Checkbox(label=T("auto_enrich"), value=False)
                bt_btn = gr.Button(T("generate"), variant="primary", size="lg")
                bt_stop = gr.Button(T("stop"), variant="stop")
                bt_log = gr.Textbox(label=T("log"), lines=8)
                bt_files = gr.Files(label=T("result"))
                ev_batch = bt_btn.click(cb_batch, [bt_text, model_dd, bt_auto], [bt_log, bt_files])
                bt_stop.click(eng.request_cancel, None, None, queue=False, cancels=[ev_batch])

    return demo


def prewarm():
    """Compilation of both paths (standard + clone) in the MAIN thread before server launch —
    first compilation in a gradio worker thread crashes the process (especially on the clone path)."""
    if eng._MOCK:
        return
    try:
        print("[prewarm] warming up + compiling in main thread (~1-3 min, one-time)...", flush=True)
        eng.generate("Warming up speech synthesis system.")
        vs = scan_voices()
        if vs:
            eng.generate("Warming up voice cloning.", ref_audio=voice_path(vs[0]))
        print("[prewarm] ready — generations will be fast and stable", flush=True)
    except Exception as e:
        print(f"[prewarm] skipped ({e})", flush=True)


if __name__ == "__main__":
    print(f"[{APP_NAME}] {DEVICE_INFO}")
    prewarm()
    build().queue(default_concurrency_limit=1).launch(
        server_port=None,
        inbrowser=(not eng._MOCK and os.environ.get("NO_AUTO_BROWSER", "").lower() not in ("1", "true", "yes")),
        i18n=I18N, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", font=["Arial", "Helvetica", "sans-serif"]),
        css=CSS, js=DARK_JS, head=HEAD_SCRIPT, show_error=True)
