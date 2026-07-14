"""Convert Nemotron Hindi-script (Devanagari) ASR output to Roman Urdu.

Nemotron has no dedicated Urdu locale, so spoken Urdu is often emitted as
Devanagari (Hindi orthography). MedRoute's symptom lexicon and demos use
**Roman Urdu** (e.g. ``mujhe bukhar hai``), so we romanize Devanagari for the
textarea and the rest of the pipeline.

English and already-roman text are left unchanged. Existing Arabic/Urdu script
is left unchanged (optional light pass-through).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# Devanagari block (Hindi / Sanskrit orthography from the ASR model)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
_HAS_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Arabic block used by Urdu (subset of Arabic script)
_HAS_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# ITRANS / IAST cleanup → clinic-style Roman Urdu (matches input_parser lexicon)
_WORD_FIXES: dict[str, str] = {
    # fever / cold / head
    "bukhaara": "bukhar",
    "bukhara": "bukhar",
    "bukhaar": "bukhar",
    "bukhar": "bukhar",
    "jukaama": "zukam",
    "jukama": "zukam",
    "jukaam": "zukam",
    "jukam": "zukam",
    "zukaama": "zukam",
    "zukama": "zukam",
    "zukam": "zukam",
    "halkaa": "halka",
    "halqa": "halka",
    "sara": "sar",
    "sira": "sir",
    "darda": "dard",
    "dina": "din",
    "aura": "aur",
    "aur": "aur",
    # body / pain
    "chaatii": "chati",
    "chaati": "chati",
    "chati": "chati",
    "seenaa": "seena",
    "seena": "seena",
    "baazuu": "bazoo",
    "baazu": "bazoo",
    "haatha": "haath",
    "haath": "haath",
    "paseenaa": "paseena",
    "paseena": "paseena",
    "pasinaa": "pasina",
    "khaansii": "khansi",
    "khaansi": "khansi",
    "khansi": "khansi",
    "saansa": "saans",
    "saans": "saans",
    "thakaana": "thakaan",
    "thakaan": "thakaan",
    "kamzorii": "kamzori",
    "ulti": "ulti",
    "ulati": "ulti",
    "dasta": "dast",
    "peta": "pet",
    "galaya": "galay",
    "galay": "galay",
    "chakkara": "chakkar",
    "chakkar": "chakkar",
    "khoona": "khoon",
    "khoon": "khoon",
    # pronouns / function words
    "mujhe": "mujhe",
    "mujhako": "mujhko",
    "mujhko": "mujhko",
    "mujhae": "mujhe",
    "hai": "hai",
    "hain": "hain",
    "hoon": "hoon",
    "hun": "hun",
    "mera": "mera",
    "meri": "meri",
    "mere": "mere",
    "aurata": "aurat",
    "dinon": "din",
    "haftaa": "hafta",
    "hafta": "hafta",
    "raata": "raat",
    "raat": "raat",
    "subaha": "subah",
    "subah": "subah",
}


def has_devanagari(text: str) -> bool:
    return bool(text and _HAS_DEVANAGARI.search(text))


def has_arabic_script(text: str) -> bool:
    return bool(text and _HAS_ARABIC.search(text))


def _itrans_to_roman_urdu(itrans: str) -> str:
    """Map ITRANS output toward clinic Roman Urdu spelling."""
    # ITRANS long vowels: A I U → aa ee oo (then lower)
    out = []
    for ch in itrans:
        if ch == "A":
            out.append("aa")
        elif ch == "I":
            out.append("ee")
        elif ch == "U":
            out.append("oo")
        else:
            out.append(ch)
    s = "".join(out).lower()
    # Normalize whitespace / punctuation spacing
    s = re.sub(r"\s+", " ", s).strip()
    # Word-level lexicon fixes
    words = re.findall(r"[a-z0-9']+|[^\s\w]", s, flags=re.UNICODE)
    fixed: list[str] = []
    for w in words:
        if re.fullmatch(r"[a-z0-9']+", w):
            fixed.append(_WORD_FIXES.get(w, _soft_schwa_strip(w)))
        else:
            fixed.append(w)
    # Re-join: no space before punctuation
    result = ""
    for w in fixed:
        if not result:
            result = w
        elif re.fullmatch(r"[^\s\w]", w):
            result += w
        else:
            result += " " + w
    return result.strip()


def _soft_schwa_strip(word: str) -> str:
    """Light trailing-schwa trim for common ASR romanizations (dina→din)."""
    if len(word) >= 4 and word.endswith("a") and not word.endswith(("aa", "ia", "ya", "na")):
        # Keep many -na/-ya words; strip bare trailing a on longer stems
        if word.endswith("na") or word.endswith("ya") or word.endswith("ra"):
            # darda→dard, dina→din, aura handled in fixes
            if word.endswith("da") or word.endswith("na") and word not in {"hona", "jana", "ana"}:
                stem = word[:-1]
                if len(stem) >= 3:
                    return stem
        elif not word.endswith(("ka", "ke", "ki", "se", "ne", "me", "ko")):
            stem = word[:-1]
            if len(stem) >= 3:
                return stem
    return word


def _transliterate_devanagari_chunk(chunk: str) -> str:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        itrans = transliterate(chunk, sanscript.DEVANAGARI, sanscript.ITRANS)
        return _itrans_to_roman_urdu(itrans)
    except Exception as exc:
        log.warning("Devanagari transliteration failed: %s", exc)
        return chunk


def to_roman_urdu(text: str) -> str:
    """Return text with Devanagari spans converted to Roman Urdu.

    Latin (English / already-roman) and Arabic/Urdu script spans are preserved.
    """
    if not text or not text.strip():
        return text
    if not has_devanagari(text):
        return text.strip()

    parts: list[str] = []
    last = 0
    for m in _DEVANAGARI_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last:m.start()])
        parts.append(_transliterate_devanagari_chunk(m.group(0)))
        last = m.end()
    if last < len(text):
        parts.append(text[last:])

    out = "".join(parts)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out).strip()
    log.info("Roman Urdu: %r → %r", text[:80], out[:80])
    return out


def prefer_clinic_transcript(text: str, language: Optional[str] = None) -> str:
    """Normalize ASR text for the MedRoute UI + symptom parser.

    - Devanagari (Hindi script from Urdu speech) → Roman Urdu
    - English / roman / Arabic Urdu → unchanged
    """
    if not text:
        return text
    lang = (language or "").lower()
    # Always romanize Devanagari when present (ur path or accidental hi)
    if has_devanagari(text):
        return to_roman_urdu(text)
    # Spoken English path or already-roman Urdu
    if lang in {"en", "en-us", "en-gb"} or not has_arabic_script(text):
        return text.strip()
    # Arabic/Urdu script: keep as-is (user can still edit; parser has roman lexicon)
    return text.strip()
