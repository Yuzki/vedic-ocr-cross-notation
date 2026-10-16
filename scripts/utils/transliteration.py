"""Transliteration: PACLIC 2026 v2 release."""

from __future__ import annotations

import re
import unicodedata

from SanskritTransliteration import transliterate as _transliterate

ACUTE: str = "^"
GRAVE: str = "_"

LEGACY_ACUTE: str = ";"
LEGACY_GRAVE: str = ":"

# Repo-local extended SLP1: ISO consonantal retroflex lateral ``ḷ`` (U+1E37) <->
# SLP1 ``L``. Upstream SanskritTransliteration collapses both ISO ``ḷ`` and ``ḍ``
# onto SLP1 ``q`` (and reverses ``q`` -> ``ḍ``), silently turning ``ḷ`` into ``ḍ``
# (study). ``L`` is unused by upstream SLP1, so we shield ``ḷ`` through upstream with
# a private-use sentinel and assign it ``L``, leaving ``ḍ`` <-> ``q`` intact.
# (Vocalic ``l̥`` stays SLP1 ``x`` -- a separate phoneme -- and is untouched here.)
_RETROFLEX_LATERAL_ISO: str = "ḷ"
_RETROFLEX_LATERAL_SLP1: str = "L"
_RETROFLEX_LATERAL_SENTINEL: str = "\ue000"  # PUA; passes upstream unchanged

# SLP1 vowel tokens that may carry an accent.
SLP1_VOWELS: str = "aAiIuUfFxXeoEO"

_VOWEL_CLASS = re.escape(SLP1_VOWELS)
_LEGACY_ACCENT_BEFORE_VOWEL = re.compile(
    rf"([{re.escape(LEGACY_ACUTE + LEGACY_GRAVE)}])(?=[{_VOWEL_CLASS}])"
)
_CANONICAL_ACCENT_BEFORE_VOWEL = re.compile(
    rf"[{re.escape(ACUTE + GRAVE)}](?=[{_VOWEL_CLASS}])"
)

# Upstream's table only carries the post-vowel diphthong rows (``aí``, ``aù`` …).
# Vedic data in this project frequently writes the accent on the first vowel of
# the diphthong (``ái``, ``àu``). Move the diacritic to the second vowel before
# delegating so the diphthong is matched as a single token.
_ISO_DIPHTHONG_NORMALISERS = (
    ("ái", "aí"),  # ái -> aí
    ("ài", "aì"),  # ài -> aì
    ("áu", "aú"),  # áu -> aú
    ("àu", "aù"),  # àu -> aù
)


def _normalise_iso_diphthongs(text: str) -> str:
    for src, dst in _ISO_DIPHTHONG_NORMALISERS:
        text = text.replace(src, dst)
    return text


# Upstream only transliterates the *decomposed* vocalic r (``r`` + combining ring
# below ``̥``), not the precomposed ISO code points ``ṛ`` (U+1E5B) / ``ṝ``
# (U+1E5D) — which NFC otherwise produces — leaving a Latin residue (study). Rewrite
# them to the decomposed spelling upstream understands (→ SLP1 ``f`` / ``F``).
# Consonantal retroflex lateral ``ḷ`` is intentionally NOT touched here: it has
# its own extended-SLP1 ``L`` contract (study), and ``ḷ`` is not always vocalic.
_ISO_VOCALIC_R_NORMALISERS = (
    ("\u1e5b", "r\u0325"),  # ṛ -> r + COMBINING RING BELOW
    ("\u1e5d", "r\u0325\u0304"),  # ṝ -> r + RING BELOW + MACRON
)


def _normalise_iso_vocalic_r(text: str) -> str:
    for src, dst in _ISO_VOCALIC_R_NORMALISERS:
        text = text.replace(src, dst)
    return text


def normalize_iso(text: str) -> str:
    """Normalize ISO 15919 text.

    Applies NFC normalisation, shifts diphthong accents from the first to the
    second vowel (``ái`` → ``aí``), and rewrites precomposed vocalic ``ṛ`` / ``ṝ``
    to their decomposed spelling so downstream tokenisation can match each as a
    single unit and upstream can transliterate the vocalic r (study).
    """
    text = _normalise_iso_diphthongs(unicodedata.normalize("NFC", text))
    return _normalise_iso_vocalic_r(text)


def _legacy_to_canonical(slp1_text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        return ACUTE if m.group(0) == LEGACY_ACUTE else GRAVE

    return _LEGACY_ACCENT_BEFORE_VOWEL.sub(_sub, slp1_text)


def _canonical_to_legacy(slp1_text: str) -> str:
    return slp1_text.replace(ACUTE, LEGACY_ACUTE).replace(GRAVE, LEGACY_GRAVE)


def normalize_slp1(text: str) -> str:
    """Canonicalize SLP1 text by translating legacy ``;``/``:`` accents to
    ``^``/``_``.

    Any ``;`` / ``:`` not directly preceding a vowel is kept untouched, so
    literal punctuation such as ``3.1.10:13.16`` survives intact.
    """
    return _legacy_to_canonical(unicodedata.normalize("NFC", text))


def strip_slp1_accents(text: str) -> str:
    """Remove acute/grave markers immediately preceding SLP1 vowels.

    Both legacy (``;``/``:``) and canonical (``^``/``_``) markers are removed.
    Other occurrences of these characters are kept untouched.
    """
    text = _legacy_to_canonical(text)
    return _CANONICAL_ACCENT_BEFORE_VOWEL.sub("", text)


def iso_to_slp1(text: str, *, accented: bool = True) -> str:
    """Transliterate ISO 15919 to SLP1.

    Args:
        text: ISO 15919 input text.
        accented: When ``True`` (default), keep acute/grave as canonical
            extended SLP1 markers (``^`` / ``_``). When ``False``, drop them
            and return plain SLP1.
    """
    text = normalize_iso(text)
    # Shield ISO ``ḷ`` through upstream (which would collapse it to ``q`` like
    # ``ḍ``) and assign it extended SLP1 ``L`` (study).
    text = text.replace(_RETROFLEX_LATERAL_ISO, _RETROFLEX_LATERAL_SENTINEL)
    out = _transliterate(text, "iso", "slp1")
    out = out.replace(_RETROFLEX_LATERAL_SENTINEL, _RETROFLEX_LATERAL_SLP1)
    if accented:
        return _legacy_to_canonical(out)
    return strip_slp1_accents(out)


def slp1_to_iso(text: str) -> str:
    """Transliterate (extended) SLP1 back to ISO 15919.

    Accepts both canonical (``^``/``_``) and legacy (``;``/``:``) accent
    markers in the input.
    """
    text = unicodedata.normalize("NFC", text)
    text = _canonical_to_legacy(text)
    # Shield extended SLP1 ``L`` through upstream (which would pass it through as a
    # literal ``L``) and restore it to ISO ``ḷ``, keeping ``q`` -> ``ḍ`` (study).
    text = text.replace(_RETROFLEX_LATERAL_SLP1, _RETROFLEX_LATERAL_SENTINEL)
    iso = _transliterate(text, "slp1", "iso")
    iso = iso.replace(_RETROFLEX_LATERAL_SENTINEL, _RETROFLEX_LATERAL_ISO)
    # SanskritTransliteration's table maps ``ḥ``/``ḫ``/``ẖ`` all to SLP1
    # ``H`` and its reverse picks the first entry (``ḫ``, jihvāmūlīya).
    # Standardize on the ISO 15919 visarga ``ḥ`` so ``iso → slp1_accented →
    # iso`` is identity and matches the corpus normalization convention used by
    # the upstream data-creation pipeline (``gretil-devanagari-synth``).
    iso = iso.replace("ḫ", "ḥ")
    return iso


def convert_text(text: str, src_scheme: str, dst_scheme: str) -> str:
    """Convert ``text`` between named text schemes.

    Supported schemes: ``iso`` (ISO 15919), ``slp1_accented`` (extended SLP1
    with ``^`` / ``_`` accents), ``slp1`` (plain SLP1, accents stripped).
    """
    src = src_scheme.strip().lower()
    dst = dst_scheme.strip().lower()
    if src == dst:
        if src == "iso":
            return normalize_iso(text)
        if src in {"slp1", "slp1_accented"}:
            return (
                normalize_slp1(text)
                if src == "slp1_accented"
                else strip_slp1_accents(text)
            )
        return text
    if src == "iso" and dst == "slp1_accented":
        return iso_to_slp1(text, accented=True)
    if src == "iso" and dst == "slp1":
        return iso_to_slp1(text, accented=False)
    if src == "slp1_accented" and dst == "iso":
        return slp1_to_iso(text)
    if src == "slp1" and dst == "iso":
        return slp1_to_iso(text)
    if src == "slp1_accented" and dst == "slp1":
        return strip_slp1_accents(text)
    if src == "slp1" and dst == "slp1_accented":
        return normalize_slp1(text)
    raise ValueError(f"unsupported scheme conversion: {src_scheme!r} -> {dst_scheme!r}")


def transliterate(text: str, input_method: str, output_method: str) -> str:
    """Backward-compatible wrapper for the previous local helper.

    Routes ISO inputs through :func:`normalize_iso` (which includes
    diphthong-accent normalisation) so legacy callers see the same
    canonical form as :func:`iso_to_slp1`. SLP1 inputs are translated from
    canonical to legacy form before delegating; SLP1 outputs are normalised
    back to canonical form. Other source/target pairs pass through to the
    upstream library unchanged.
    """
    src = input_method.strip().lower()
    dst = output_method.strip().lower()
    payload = text
    if src == "iso":
        payload = normalize_iso(payload)
        # Mirror ``iso_to_slp1``'s retroflex-lateral protection for the
        # iso→slp1 path only (other targets keep upstream behaviour) (study).
        if dst == "slp1":
            payload = payload.replace(
                _RETROFLEX_LATERAL_ISO, _RETROFLEX_LATERAL_SENTINEL
            )
    elif src == "slp1":
        payload = _canonical_to_legacy(unicodedata.normalize("NFC", payload))
        if dst == "iso":
            payload = payload.replace(
                _RETROFLEX_LATERAL_SLP1, _RETROFLEX_LATERAL_SENTINEL
            )
    out = _transliterate(payload, input_method, output_method)
    if dst == "slp1":
        if src == "iso":
            out = out.replace(_RETROFLEX_LATERAL_SENTINEL, _RETROFLEX_LATERAL_SLP1)
        out = _legacy_to_canonical(out)
    elif dst == "iso":
        if src == "slp1":
            out = out.replace(_RETROFLEX_LATERAL_SENTINEL, _RETROFLEX_LATERAL_ISO)
        # Share the visarga normalisation with ``slp1_to_iso`` so legacy
        # callers don't observe a different ``H`` → ``ḫ`` (U+1E2B) mapping
        # than the production path.
        out = out.replace("ḫ", "ḥ")
    return out


if __name__ == "__main__":
    sample = "agnim īḷe puróhitam"
    slp1 = iso_to_slp1(sample)
    print(f"ISO  : {sample}")
    print(f"SLP1 : {slp1}")
    print(f"back : {slp1_to_iso(slp1)}")
