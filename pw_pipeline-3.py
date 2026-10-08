#!/usr/bin/env python3
"""
pw_pipeline.py

End-to-end spoken-audio PII redaction pipeline.

Workflow
--------
1. Standardize the input audio to 16 kHz without modifying the original.
2. Use the language-ID/transcription logic from asr_li.py.
3. Use ctc_score.transcribe_with_timestamps() for word timestamps.
4. Use qwen_ner.detect_pii() for multilingual PII span detection.
5. Cross-check spoken email spans with ned_1.detect_spoken_email().
6. Align transcript character spans to CTC word timestamps.
7. Merge overlapping redaction intervals.
8. Replace each interval with repeated/trimmed beep audio.
9. Save <original_stem>_<original_suffix> at 16 kHz and write a JSON report.

Example
-------
    python pw_pipeline.py recording.wav --beep beep.wav

Notes
-----
- asr_li.py currently has no callable function and executes inference at import time,
  so its model-loading/transcription logic is adapted here into a lazy helper.
- ctc_score.py already exposes transcribe_with_timestamps(audio_file, language),
  so this script calls it directly.
- qwen_ner.py already exposes detect_pii(transcript), so it is called directly.
- ned_1.py exposes detect_spoken_email(text); its example print is suppressed on import.
- audio_beep.py has reusable beep logic but also executes a hard-coded example on import,
  so the same make/repeat/trim behavior is implemented here without importing the file.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydub import AudioSegment


TARGET_SAMPLE_RATE = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent

# qwen_ner.py calls this PHONE_NUMBER. The user-facing report uses the requested
# category name "Number" while preserving raw_type as well.
CATEGORY_MAP = {
    "NAME": "Name",
    "EMAIL": "Email",
    "PHONE_NUMBER": "Number",
    "NUMBER": "Number",
    "PAN": "PAN",
    "AADHAAR": "Aadhaar",
    "ADDRESS": "Address",
}

LANGUAGE_ALIASES = {
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "ta": "ta",
    "tam": "ta",
    "tamil": "ta",
    "bn": "bn",
    "ben": "bn",
    "bengali": "bn",
    "bangla": "bn",
    "te": "te",
    "tel": "te",
    "telugu": "te",
    "kn": "kn",
    "kan": "kn",
    "kannada": "kn",
    "ml": "ml",
    "mal": "ml",
    "malayalam": "ml",
    "gu": "gu",
    "guj": "gu",
    "gujarati": "gu",
}

SUPPORTED_LANGUAGES = tuple(sorted(set(LANGUAGE_ALIASES.values())))


@dataclass
class TextToken:
    index: int
    text: str
    start: int
    end: int
    norm: str


@dataclass
class AlignedEntity:
    category: str
    raw_type: str
    text: str
    transcript_start: int
    transcript_end: int
    audio_start: float
    audio_end: float
    source: List[str]
    alignment_method: str
    validation: Optional[str] = None
    cross_check: Optional[str] = None


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _normalize_language_id(lid: Any) -> str:
    """Normalize language-ID output to one of hi/ta/bn/te/kn/ml/gu."""
    if lid is None:
        raise ValueError("Language identification returned None.")

    # Some libraries may return dict-like results.
    if isinstance(lid, dict):
        for key in ("language", "lang", "lid", "code", "label"):
            if key in lid:
                lid = lid[key]
                break

    value = str(lid).strip().lower()
    value = value.replace("_", "-")

    # Accept values such as hi-IN, hin_IN, __label__hi, language=Hindi.
    value = value.replace("__label__", "")
    for sep in ("=", ":"):
        if sep in value:
            value = value.split(sep)[-1].strip()

    if value in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[value]

    primary = value.split("-")[0]
    if primary in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[primary]

    # Last chance: match a language name embedded in a descriptive string.
    for alias, code in LANGUAGE_ALIASES.items():
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", value):
            return code

    raise ValueError(
        f"Unsupported detected language ID {lid!r}. "
        f"Supported pipeline languages: {', '.join(SUPPORTED_LANGUAGES)}"
    )


def _normalize_token(text: str) -> str:
    """
    Normalize a token for transcript/CTC matching while preserving Indic letters,
    combining marks and digits. Punctuation and spacing are ignored only for
    matching; original text/offsets remain untouched.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    chars: List[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category and category[0] in {"L", "M", "N"}:
            chars.append(ch)
    return "".join(chars)


def _tokenize_with_spans(text: str) -> List[TextToken]:
    tokens: List[TextToken] = []
    for match in re.finditer(r"\S+", text, flags=re.UNICODE):
        raw = match.group(0)
        tokens.append(
            TextToken(
                index=len(tokens),
                text=raw,
                start=match.start(),
                end=match.end(),
                norm=_normalize_token(raw),
            )
        )
    return tokens


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


# -----------------------------------------------------------------------------
# Audio standardization
# -----------------------------------------------------------------------------


def prepare_16k_audio(input_path: Path, keep_resampled: bool = False) -> Dict[str, Any]:
    """
    Return metadata and the exact audio path to use for ALL downstream stages.

    If already 16 kHz, the original path is used directly. Otherwise pydub performs
    a real resampling operation and writes a temporary 16 kHz WAV. The source file
    is never modified.
    """
    audio = AudioSegment.from_file(str(input_path))
    original_sr = int(audio.frame_rate)
    original_channels = int(audio.channels)
    original_duration_ms = len(audio)

    if original_sr == TARGET_SAMPLE_RATE:
        logging.info("Input is already 16 kHz; using it directly.")
        return {
            "path": input_path,
            "temporary": False,
            "resampled": False,
            "original_sample_rate": original_sr,
            "processing_sample_rate": TARGET_SAMPLE_RATE,
            "channels": original_channels,
            "duration_ms": original_duration_ms,
            "cleanup": False,
        }

    logging.info("Resampling input from %d Hz to 16000 Hz.", original_sr)

    # AudioSegment.set_frame_rate performs sample-rate conversion; it does not
    # merely rewrite the sample-rate metadata.
    resampled = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    if abs(len(resampled) - original_duration_ms) > 2:
        raise RuntimeError(
            "Resampling unexpectedly changed audio duration: "
            f"{original_duration_ms} ms -> {len(resampled)} ms"
        )

    if keep_resampled:
        work_path = input_path.with_name(f"{input_path.stem}.pw_16k.wav")
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="pw_pipeline_", suffix="_16k.wav", delete=False)
        work_path = Path(tmp.name)
        tmp.close()

    resampled.export(str(work_path), format="wav")

    return {
        "path": work_path,
        "temporary": not keep_resampled,
        "resampled": True,
        "original_sample_rate": original_sr,
        "processing_sample_rate": TARGET_SAMPLE_RATE,
        "channels": int(resampled.channels),
        "duration_ms": len(resampled),
        "cleanup": not keep_resampled,
    }


# -----------------------------------------------------------------------------
# asr_li.py adapter
# -----------------------------------------------------------------------------


_ASR_LI_MODEL = None


def identify_language_and_transcribe(audio_path: Path) -> Tuple[str, str, Any]:
    """
    Adapt the actual asr_li.py logic into a callable helper.

    asr_li.py currently does:
      snapshot_download("bodhan-ai/Indic-Transcribe-Flex")
      IndicTranscribe.from_pretrained(model_dir)
      asr.transcribe(path, return_lid=True)

    Returns:
      normalized language code, transcript, raw language-ID result
    """
    global _ASR_LI_MODEL

    if _ASR_LI_MODEL is None:
        logging.info("Loading language-ID / transcription model from asr_li.py logic...")
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download("bodhan-ai/Indic-Transcribe-Flex")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from indic_transcribe import IndicTranscribe

        _ASR_LI_MODEL = IndicTranscribe.from_pretrained(model_dir)

    transcript, raw_lid = _ASR_LI_MODEL.transcribe(
        str(audio_path),
        return_lid=True,
    )

    transcript = str(transcript).strip()
    language = _normalize_language_id(raw_lid)

    if not transcript:
        raise RuntimeError("asr_li transcription returned an empty transcript.")

    logging.info("Detected language: %s (raw LID=%r)", language, raw_lid)
    return language, transcript, raw_lid


# -----------------------------------------------------------------------------
# ctc_score.py integration
# -----------------------------------------------------------------------------


def get_ctc_word_timestamps(audio_path: Path, language: str) -> Dict[str, Any]:
    """Call the actual ctc_score.transcribe_with_timestamps() interface."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    ctc_score = importlib.import_module("ctc_score")

    if not hasattr(ctc_score, "transcribe_with_timestamps"):
        raise AttributeError(
            "ctc_score.py does not expose transcribe_with_timestamps(audio_file, language)."
        )

    result = ctc_score.transcribe_with_timestamps(
        str(audio_path),
        language,
    )

    if not isinstance(result, dict):
        raise TypeError("ctc_score.transcribe_with_timestamps() must return a dict.")

    words = result.get("words")
    if not isinstance(words, list) or not words:
        raise RuntimeError("ctc_score.py returned no word timestamps.")

    normalized_words: List[Dict[str, Any]] = []
    for i, word in enumerate(words):
        try:
            text = str(word["word"])
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CTC word entry at index {i}: {word!r}") from exc

        if end < start:
            raise ValueError(f"CTC word has end < start at index {i}: {word!r}")

        normalized_words.append(
            {
                "index": i,
                "word": text,
                "start": start,
                "end": end,
                "norm": _normalize_token(text),
            }
        )

    return {
        "language": result.get("language", language),
        "transcript": str(result.get("transcript", "")),
        "words": normalized_words,
    }


# -----------------------------------------------------------------------------
# qwen_ner.py integration
# -----------------------------------------------------------------------------


def detect_qwen_entities(transcript: str) -> List[Dict[str, Any]]:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    logging.info("Loading/running qwen_ner.py...")
    qwen_ner = importlib.import_module("qwen_ner")

    if not hasattr(qwen_ner, "detect_pii"):
        raise AttributeError("qwen_ner.py does not expose detect_pii(transcript).")

    raw_entities = qwen_ner.detect_pii(transcript)
    if not isinstance(raw_entities, list):
        raise TypeError("qwen_ner.detect_pii() did not return a list.")

    entities: List[Dict[str, Any]] = []
    for i, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            logging.warning("Ignoring non-dict Qwen entity at index %d: %r", i, entity)
            continue

        raw_type = str(entity.get("type", "")).upper().strip()
        if raw_type not in CATEGORY_MAP:
            logging.warning("Ignoring unsupported Qwen entity type %r", raw_type)
            continue

        text = entity.get("text")
        start = entity.get("start")
        end = entity.get("end")

        if not isinstance(text, str) or not isinstance(start, int) or not isinstance(end, int):
            logging.warning("Ignoring Qwen entity with invalid span: %r", entity)
            continue

        # qwen_ner.py is already designed to return exact substrings. Verify again.
        if start < 0 or end > len(transcript) or start >= end or transcript[start:end] != text:
            logging.warning("Rejecting Qwen entity that is not an exact transcript span: %r", entity)
            continue

        entities.append(
            {
                "category": CATEGORY_MAP[raw_type],
                "raw_type": raw_type,
                "text": text,
                "start": start,
                "end": end,
                "source": ["qwen_ner"],
                "validation": entity.get("validation"),
                "cross_check": None,
            }
        )

    return entities


# -----------------------------------------------------------------------------
# ned_1.py cross-check
# -----------------------------------------------------------------------------


def detect_ned_entities(transcript: str) -> List[Dict[str, Any]]:
    """
    ned_1.py currently only cross-checks spoken EMAIL forms. Its module-level
    example prints on import, so suppress that incidental output.
    """
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    with contextlib.redirect_stdout(io.StringIO()):
        ned_1 = importlib.import_module("ned_1")

    if not hasattr(ned_1, "detect_spoken_email"):
        raise AttributeError("ned_1.py does not expose detect_spoken_email(text).")

    raw = ned_1.detect_spoken_email(transcript)
    if not isinstance(raw, list):
        raise TypeError("ned_1.detect_spoken_email() did not return a list.")

    results: List[Dict[str, Any]] = []
    for entity in raw:
        try:
            text = str(entity["text"])
            start = int(entity["start"])
            end = int(entity["end"])
        except (KeyError, TypeError, ValueError):
            logging.warning("Ignoring malformed ned_1 entity: %r", entity)
            continue

        if start < 0 or end > len(transcript) or start >= end:
            continue
        if transcript[start:end] != text:
            continue

        results.append(
            {
                "category": "Email",
                "raw_type": "EMAIL",
                "text": text,
                "start": start,
                "end": end,
                "source": ["ned_1"],
                "confidence": entity.get("confidence"),
                "cross_check": "ned_1_only",
            }
        )

    return results


def reconcile_entities(
    transcript: str,
    qwen_entities: List[Dict[str, Any]],
    ned_entities: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reconcile qwen_ner with ned_1.

    ned_1 only checks spoken emails, so non-email Qwen entities are not considered
    disagreements. For emails:
      - exact same span: mark agreed
      - overlapping but different boundaries: use the union span conservatively,
        keep both sources, and report a boundary disagreement
      - NED-only email: retain it and report it
      - Qwen-only email: retain it and report it
    """
    final_entities = [dict(e) for e in qwen_entities]
    disagreements: List[Dict[str, Any]] = []
    matched_qwen_email_indices: set[int] = set()

    for ned in ned_entities:
        exact_idx: Optional[int] = None
        overlap_idx: Optional[int] = None

        for i, q in enumerate(final_entities):
            if q["category"] != "Email":
                continue
            if q["start"] == ned["start"] and q["end"] == ned["end"]:
                exact_idx = i
                break
            if _ranges_overlap(q["start"], q["end"], ned["start"], ned["end"]):
                overlap_idx = i

        if exact_idx is not None:
            q = final_entities[exact_idx]
            q["source"] = sorted(set(q.get("source", [])) | {"ned_1"})
            q["cross_check"] = "agreed"
            matched_qwen_email_indices.add(exact_idx)
            continue

        if overlap_idx is not None:
            q = final_entities[overlap_idx]
            original_qwen = {
                "text": q["text"],
                "start": q["start"],
                "end": q["end"],
            }
            merged_start = min(q["start"], ned["start"])
            merged_end = max(q["end"], ned["end"])
            q["start"] = merged_start
            q["end"] = merged_end
            q["text"] = transcript[merged_start:merged_end]
            q["source"] = sorted(set(q.get("source", [])) | {"ned_1"})
            q["cross_check"] = "boundary_disagreement_merged"
            matched_qwen_email_indices.add(overlap_idx)
            disagreements.append(
                {
                    "kind": "email_boundary_disagreement",
                    "qwen": original_qwen,
                    "ned_1": {
                        "text": ned["text"],
                        "start": ned["start"],
                        "end": ned["end"],
                    },
                    "resolution": {
                        "text": q["text"],
                        "start": merged_start,
                        "end": merged_end,
                        "policy": "union span for conservative redaction",
                    },
                }
            )
            continue

        final_entities.append(dict(ned))
        disagreements.append(
            {
                "kind": "ned_1_only_email",
                "ned_1": {
                    "text": ned["text"],
                    "start": ned["start"],
                    "end": ned["end"],
                },
                "resolution": "retained as additional sensitive entity",
            }
        )

    for i, q in enumerate(final_entities):
        if q.get("category") == "Email" and "qwen_ner" in q.get("source", []):
            if i not in matched_qwen_email_indices and "ned_1" not in q.get("source", []):
                q["cross_check"] = q.get("cross_check") or "qwen_only"
                disagreements.append(
                    {
                        "kind": "qwen_only_email",
                        "qwen": {
                            "text": q["text"],
                            "start": q["start"],
                            "end": q["end"],
                        },
                        "note": "ned_1 only recognizes its configured spoken-email patterns; Qwen entity retained.",
                    }
                )

    # Deduplicate exact category+span duplicates, merging sources.
    dedup: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for entity in final_entities:
        key = (entity["category"], entity["start"], entity["end"])
        if key not in dedup:
            dedup[key] = dict(entity)
        else:
            dedup[key]["source"] = sorted(
                set(dedup[key].get("source", [])) | set(entity.get("source", []))
            )

    final_entities = sorted(dedup.values(), key=lambda x: (x["start"], x["end"], x["category"]))
    return final_entities, disagreements


# -----------------------------------------------------------------------------
# Transcript -> CTC token alignment
# -----------------------------------------------------------------------------


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def build_transcript_to_ctc_map(
    transcript: str,
    ctc_words: Sequence[Dict[str, Any]],
    fuzzy_threshold: float = 0.62,
) -> Tuple[List[TextToken], Dict[int, int], Dict[str, Any]]:
    """
    Build a monotonic occurrence-aware mapping from canonical transcript tokens to
    CTC word indices. Sequence order, rather than first string occurrence, is what
    disambiguates repeated words/entities.
    """
    transcript_tokens = _tokenize_with_spans(transcript)
    a = [t.norm for t in transcript_tokens]
    b = [str(w.get("norm") or _normalize_token(str(w.get("word", "")))) for w in ctc_words]

    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    mapping: Dict[int, int] = {}
    operations: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        operations.append({"tag": tag, "a": [i1, i2], "b": [j1, j2]})

        if tag == "equal":
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1 + offset
            continue

        if tag != "replace":
            continue

        # Fuzzy, order-preserving matching inside replacement blocks.
        last_j = j1 - 1
        for i in range(i1, i2):
            best_j: Optional[int] = None
            best_score = 0.0
            for j in range(max(j1, last_j + 1), j2):
                score = _token_similarity(a[i], b[j])
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None and best_score >= fuzzy_threshold:
                mapping[i] = best_j
                last_j = best_j

    mapped_count = len(mapping)
    stats = {
        "transcript_token_count": len(transcript_tokens),
        "ctc_word_count": len(ctc_words),
        "mapped_token_count": mapped_count,
        "mapped_fraction": mapped_count / len(transcript_tokens) if transcript_tokens else 0.0,
        "opcodes": operations,
    }
    return transcript_tokens, mapping, stats


def _entity_token_indices(entity: Dict[str, Any], tokens: Sequence[TextToken]) -> List[int]:
    indices: List[int] = []
    start = int(entity["start"])
    end = int(entity["end"])
    for token in tokens:
        if not token.norm:
            continue
        if _ranges_overlap(start, end, token.start, token.end):
            indices.append(token.index)
    return indices


def _position_aware_entity_fallback(
    entity_tokens: Sequence[TextToken],
    ctc_words: Sequence[Dict[str, Any]],
    entity_mid_char: float,
    transcript_length: int,
    fuzzy_threshold: float,
) -> Optional[Tuple[int, int, float]]:
    """
    Find a contiguous CTC n-gram corresponding to an entity. If repeated candidates
    exist, choose using both textual similarity and expected transcript position.
    """
    needle = [t.norm for t in entity_tokens if t.norm]
    if not needle or len(ctc_words) < len(needle):
        return None

    expected_center = (
        (entity_mid_char / max(1, transcript_length)) * max(0, len(ctc_words) - 1)
    )

    candidates: List[Tuple[float, float, int, int]] = []
    n = len(needle)
    for start in range(0, len(ctc_words) - n + 1):
        scores = []
        for offset, token in enumerate(needle):
            ctc_norm = str(ctc_words[start + offset].get("norm", ""))
            scores.append(_token_similarity(token, ctc_norm))
        text_score = sum(scores) / len(scores)
        if text_score < fuzzy_threshold:
            continue

        end = start + n - 1
        center = (start + end) / 2.0
        position_distance = abs(center - expected_center) / max(1, len(ctc_words))

        # Prioritize transcript similarity, then occurrence position.
        candidates.append((text_score, -position_distance, start, end))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    score, _neg_distance, start, end = candidates[0]
    return start, end, score


def align_entities_to_audio(
    transcript: str,
    entities: Sequence[Dict[str, Any]],
    ctc_words: Sequence[Dict[str, Any]],
    fuzzy_threshold: float = 0.62,
) -> Tuple[List[AlignedEntity], List[Dict[str, Any]], Dict[str, Any]]:
    transcript_tokens, token_map, map_stats = build_transcript_to_ctc_map(
        transcript,
        ctc_words,
        fuzzy_threshold=fuzzy_threshold,
    )

    aligned: List[AlignedEntity] = []
    unaligned: List[Dict[str, Any]] = []

    for entity in entities:
        token_indices = _entity_token_indices(entity, transcript_tokens)
        if not token_indices:
            unaligned.append(
                {
                    **entity,
                    "reason": "entity span does not overlap any non-punctuation transcript token",
                }
            )
            continue

        first_token_idx = token_indices[0]
        last_token_idx = token_indices[-1]

        first_ctc = token_map.get(first_token_idx)
        last_ctc = token_map.get(last_token_idx)
        method = "global_sequence_alignment"

        if first_ctc is None or last_ctc is None or last_ctc < first_ctc:
            entity_tokens = [transcript_tokens[i] for i in token_indices]
            fallback = _position_aware_entity_fallback(
                entity_tokens=entity_tokens,
                ctc_words=ctc_words,
                entity_mid_char=(entity["start"] + entity["end"]) / 2.0,
                transcript_length=len(transcript),
                fuzzy_threshold=max(fuzzy_threshold, 0.70),
            )
            if fallback is None:
                unaligned.append(
                    {
                        **entity,
                        "reason": "could not map first/last entity token to CTC word timestamps",
                        "first_transcript_token": transcript_tokens[first_token_idx].text,
                        "last_transcript_token": transcript_tokens[last_token_idx].text,
                    }
                )
                continue

            first_ctc, last_ctc, fallback_score = fallback
            method = f"position_aware_ngram_fallback(score={fallback_score:.3f})"

        try:
            audio_start = float(ctc_words[first_ctc]["start"])
            audio_end = float(ctc_words[last_ctc]["end"])
        except (IndexError, KeyError, TypeError, ValueError):
            unaligned.append({**entity, "reason": "invalid mapped CTC timestamp entry"})
            continue

        if audio_end <= audio_start:
            unaligned.append({**entity, "reason": "mapped audio interval has non-positive duration"})
            continue

        aligned.append(
            AlignedEntity(
                category=str(entity["category"]),
                raw_type=str(entity.get("raw_type", entity["category"])),
                text=str(entity["text"]),
                transcript_start=int(entity["start"]),
                transcript_end=int(entity["end"]),
                audio_start=audio_start,
                audio_end=audio_end,
                source=list(entity.get("source", [])),
                alignment_method=method,
                validation=entity.get("validation"),
                cross_check=entity.get("cross_check"),
            )
        )

    aligned.sort(key=lambda e: (e.audio_start, e.audio_end))
    return aligned, unaligned, map_stats


# -----------------------------------------------------------------------------
# Interval merge + beep replacement
# -----------------------------------------------------------------------------


def merge_redaction_intervals(
    aligned_entities: Sequence[AlignedEntity],
    padding_ms: int = 0,
    audio_duration_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []
    pad_sec = max(0, padding_ms) / 1000.0
    max_sec = audio_duration_ms / 1000.0 if audio_duration_ms is not None else None

    for entity in aligned_entities:
        start = max(0.0, entity.audio_start - pad_sec)
        end = entity.audio_end + pad_sec
        if max_sec is not None:
            end = min(end, max_sec)
        if end <= start:
            continue
        intervals.append(
            {
                "start": start,
                "end": end,
                "entities": [
                    {
                        "category": entity.category,
                        "text": entity.text,
                        "transcript_start": entity.transcript_start,
                        "transcript_end": entity.transcript_end,
                    }
                ],
            }
        )

    intervals.sort(key=lambda x: (x["start"], x["end"]))
    merged: List[Dict[str, Any]] = []

    for interval in intervals:
        if not merged or interval["start"] > merged[-1]["end"]:
            merged.append(interval)
            continue

        merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        merged[-1]["entities"].extend(interval["entities"])

    return merged


def _make_beep_segment(beep: AudioSegment, duration_ms: int) -> AudioSegment:
    """Same repeat-then-trim behavior as audio_beep.make_beep_segment()."""
    if duration_ms <= 0:
        return beep[:0]
    if len(beep) <= 0:
        raise ValueError("beep.wav is empty.")
    repeats = (duration_ms // len(beep)) + 1
    return (beep * repeats)[:duration_ms]


def replace_intervals_with_beep(
    input_audio: Path,
    beep_audio: Path,
    output_audio: Path,
    intervals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Replace sensitive intervals with repeated/trimmed beep while preserving duration.
    The beep is genuinely resampled to 16 kHz and matched to the processed audio's
    channel count/sample width.
    """
    audio = AudioSegment.from_file(str(input_audio))
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Processed audio must be 16 kHz before redaction; got {audio.frame_rate} Hz."
        )

    beep = AudioSegment.from_file(str(beep_audio))
    beep_original_sr = int(beep.frame_rate)
    beep_original_channels = int(beep.channels)

    beep = (
        beep.set_frame_rate(TARGET_SAMPLE_RATE)
        .set_channels(audio.channels)
        .set_sample_width(audio.sample_width)
    )

    original_duration_ms = len(audio)
    result = audio

    # Intervals are already merged and sorted. Equal-duration replacement means
    # timestamps remain stable after every edit.
    for interval in intervals:
        start_ms = max(0, int(round(float(interval["start"]) * 1000.0)))
        end_ms = min(len(result), int(round(float(interval["end"]) * 1000.0)))

        if end_ms <= start_ms:
            logging.warning("Skipping empty redaction interval %.3f-%.3f", interval["start"], interval["end"])
            continue

        replacement = _make_beep_segment(beep, end_ms - start_ms)
        result = result[:start_ms] + replacement + result[end_ms:]

    if len(result) != original_duration_ms:
        raise RuntimeError(
            f"Redaction changed audio duration: {original_duration_ms} ms -> {len(result)} ms"
        )

    # Defensive: guarantee the output is 16 kHz and has the processed channel count.
    result = result.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(audio.channels)

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_audio.suffix.lower().lstrip(".") or "wav"
    result.export(str(output_audio), format=output_format)

    return {
        "output": str(output_audio),
        "sample_rate": int(result.frame_rate),
        "channels": int(result.channels),
        "duration_ms": len(result),
        "beep_original_sample_rate": beep_original_sr,
        "beep_original_channels": beep_original_channels,
        "beep_processing_sample_rate": TARGET_SAMPLE_RATE,
        "beep_processing_channels": int(beep.channels),
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".wav"
    return input_path.with_name(f"{input_path.stem}_{suffix}")


def default_report_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_redaction_report.json")


def run_pipeline(
    input_audio: Path,
    beep_audio: Path,
    report_path: Optional[Path] = None,
    keep_resampled: bool = False,
    padding_ms: int = 0,
    fuzzy_threshold: float = 0.62,
) -> Dict[str, Any]:
    input_audio = input_audio.resolve()
    beep_audio = beep_audio.resolve()

    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")
    if not beep_audio.exists():
        raise FileNotFoundError(f"Beep audio not found: {beep_audio}")

    output_audio = default_output_path(input_audio)
    report_path = (report_path or default_report_path(input_audio)).resolve()

    if output_audio.resolve() == input_audio.resolve():
        raise RuntimeError("Output path unexpectedly resolves to the input path.")

    processing_info = prepare_16k_audio(input_audio, keep_resampled=keep_resampled)
    work_audio = Path(processing_info["path"])

    report: Dict[str, Any] = {
        "input_audio": str(input_audio),
        "output_audio": str(output_audio),
        "beep_audio": str(beep_audio),
        "audio": {
            "original_sample_rate": processing_info["original_sample_rate"],
            "processing_sample_rate": processing_info["processing_sample_rate"],
            "channels": processing_info["channels"],
            "duration_ms": processing_info["duration_ms"],
            "resampled": processing_info["resampled"],
        },
        "language": None,
        "raw_language_id": None,
        "transcript": None,
        "ctc_transcript": None,
        "ctc_transcript_exact_match": None,
        "qwen_entities": [],
        "ned_1_entities": [],
        "cross_check_disagreements": [],
        "final_entities": [],
        "aligned_entities": [],
        "unaligned_entities": [],
        "alignment_stats": {},
        "redaction_intervals": [],
        "output_audio_info": {},
    }

    try:
        # 1. Language ID + canonical transcript.
        logging.info("[1/6] Language identification and transcription")
        language, transcript, raw_lid = identify_language_and_transcribe(work_audio)
        report["language"] = language
        report["raw_language_id"] = str(raw_lid)
        report["transcript"] = transcript

        # 2. Word-level timestamps from ctc_score.py.
        logging.info("[2/6] CTC word timestamps")
        ctc_result = get_ctc_word_timestamps(work_audio, language)
        report["ctc_transcript"] = ctc_result["transcript"]
        report["ctc_transcript_exact_match"] = ctc_result["transcript"].strip() == transcript.strip()

        if not report["ctc_transcript_exact_match"]:
            logging.warning(
                "asr_li transcript and ctc_score transcript differ; using occurrence-aware sequence alignment."
            )

        # 3. Qwen PII detection.
        logging.info("[3/6] Qwen sensitive-entity detection")
        qwen_entities = detect_qwen_entities(transcript)
        report["qwen_entities"] = qwen_entities

        # 4. ned_1 cross-check.
        logging.info("[4/6] Cross-check with ned_1.py")
        ned_entities = detect_ned_entities(transcript)
        report["ned_1_entities"] = ned_entities

        final_entities, disagreements = reconcile_entities(
            transcript,
            qwen_entities,
            ned_entities,
        )
        report["final_entities"] = final_entities
        report["cross_check_disagreements"] = disagreements

        if disagreements:
            logging.warning("Cross-check found %d disagreement(s).", len(disagreements))
            for disagreement in disagreements:
                logging.warning("  %s", json.dumps(disagreement, ensure_ascii=False))

        # 5. Map transcript spans to audio timestamps.
        logging.info("[5/6] Mapping sensitive spans to CTC timestamps")
        aligned, unaligned, alignment_stats = align_entities_to_audio(
            transcript=transcript,
            entities=final_entities,
            ctc_words=ctc_result["words"],
            fuzzy_threshold=fuzzy_threshold,
        )
        report["aligned_entities"] = [asdict(e) for e in aligned]
        report["unaligned_entities"] = unaligned
        report["alignment_stats"] = alignment_stats

        if unaligned:
            logging.warning("%d sensitive entity/entities could not be aligned:", len(unaligned))
            for entity in unaligned:
                logging.warning(
                    "  %s [%s:%s] reason=%s",
                    entity.get("text"),
                    entity.get("start"),
                    entity.get("end"),
                    entity.get("reason"),
                )

        intervals = merge_redaction_intervals(
            aligned,
            padding_ms=padding_ms,
            audio_duration_ms=processing_info["duration_ms"],
        )
        report["redaction_intervals"] = intervals

        # 6. Beep replacement.
        logging.info("[6/6] Replacing %d merged interval(s) with beep", len(intervals))
        output_info = replace_intervals_with_beep(
            input_audio=work_audio,
            beep_audio=beep_audio,
            output_audio=output_audio,
            intervals=intervals,
        )
        report["output_audio_info"] = output_info

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info("Redacted audio: %s", output_audio)
        logging.info("Report: %s", report_path)

        return report

    finally:
        if processing_info.get("cleanup") and work_audio.exists():
            try:
                work_audio.unlink()
            except OSError as exc:
                logging.warning("Could not remove temporary 16 kHz file %s: %s", work_audio, exc)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify multilingual spoken PII, map it to CTC word timestamps, "
            "and replace the corresponding audio with a beep."
        )
    )
    parser.add_argument(
        "input_audio",
        type=Path,
        help="Input audio file. The original file is never modified.",
    )
    parser.add_argument(
        "--beep",
        type=Path,
        default=SCRIPT_DIR / "beep.wav",
        help="Beep audio file (default: beep.wav beside pw_pipeline.py).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <input_stem>_redaction_report.json).",
    )
    parser.add_argument(
        "--padding-ms",
        type=int,
        default=0,
        help="Optional extra redaction padding before/after each aligned entity (default: 0).",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=0.62,
        help="Minimum fuzzy transcript-to-CTC token similarity in [0,1] (default: 0.62).",
    )
    parser.add_argument(
        "--keep-16k",
        action="store_true",
        help="Keep the intermediate 16 kHz WAV when resampling was required.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if not 0.0 <= args.alignment_threshold <= 1.0:
        parser.error("--alignment-threshold must be between 0 and 1.")
    if args.padding_ms < 0:
        parser.error("--padding-ms must be >= 0.")

    try:
        report = run_pipeline(
            input_audio=args.input_audio,
            beep_audio=args.beep,
            report_path=args.report,
            keep_resampled=args.keep_16k,
            padding_ms=args.padding_ms,
            fuzzy_threshold=args.alignment_threshold,
        )
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1

    print("\n=== PII REDACTION COMPLETE ===")
    print(f"Language:      {report['language']}")
    print(f"Transcript:    {report['transcript']}")
    print(f"Entities:      {len(report['final_entities'])}")
    print(f"Aligned:       {len(report['aligned_entities'])}")
    print(f"Unaligned:     {len(report['unaligned_entities'])}")
    print(f"Disagreements: {len(report['cross_check_disagreements'])}")
    print(f"Intervals:     {len(report['redaction_intervals'])}")
    print(f"Output:        {report['output_audio']}")
    print(f"Report:        {args.report or default_report_path(args.input_audio.resolve())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
