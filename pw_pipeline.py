"""
PII audio redaction pipeline.

    audio
      -> resample to 16 kHz mono WAV if needed (<name>_16k.wav)
      -> Silero VAD: temporary speech segments
      -> asr_li.py     : language identification + transcription
      -> ctc_score.py   : word-level timestamps
      -> qwen_ner.py    : NAME / EMAIL / PHONE_NUMBER / PAN / AADHAAR / ADDRESS
      -> ned_1.py       : cross-check (spoken e-mail detection)
      -> map PII words to original full-audio timestamps
      -> audio_beep.py  : replace PII audio with beep.wav
      -> <name>_.<ext>

Usage:
    python pw_pipeline.py input.wav
    python pw_pipeline.py input.wav --beep beep.wav --lang hi --pad 0.1
"""

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import logging
import shutil
import traceback
from collections import Counter
from datetime import datetime, timezone
import unicodedata
from difflib import SequenceMatcher
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

# Helper scripts may live next to this file or in ./extra
HELPER_DIRS = [HERE, os.path.join(HERE, "extra")]

for _d in reversed(HELPER_DIRS):
    sys.path.insert(0, _d)


def helper_path(filename: str) -> str:
    for d in HELPER_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found in {HELPER_DIRS}")


# ============================================================
# Helper-module loading
# ============================================================

def load_definitions(filename: str) -> ModuleType:
    """
    Load only imports, functions, classes and assignments from a helper
    script, skipping its top-level example code (audio_beep.py and
    ned_1.py run their examples on import).
    """

    path = helper_path(filename)

    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    keep = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
    )

    tree.body = [
        node for node in tree.body
        if isinstance(node, keep)
        # skip example assignments that call functions, e.g. intervals = [...]
        and not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in {"intervals", "text"}
                for t in node.targets
            )
        )
    ]

    module = ModuleType(os.path.splitext(filename)[0])
    module.__file__ = path
    exec(compile(tree, path, "exec"), module.__dict__)

    return module


# ============================================================
# 1. Language identification + transcription (asr_li.py)
# ============================================================

_INDIC_ASR = None


def identify_and_transcribe(audio_file: str) -> Tuple[str, str]:
    """
    Same calls as asr_li.py (which hard-codes its input file and so
    cannot be imported directly).
    """

    global _INDIC_ASR

    if _INDIC_ASR is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download("bodhan-ai/Indic-Transcribe-Flex")
        # The snapshot contains a nemo/ directory of checkpoint helpers. Leaving
        # it on sys.path makes `import nemo` load that namespace instead of NeMo.
        sys.path.insert(0, model_dir)
        try:
            from indic_transcribe import IndicTranscribe
            _INDIC_ASR = IndicTranscribe.from_pretrained(model_dir)
        finally:
            sys.path.remove(model_dir)

    text, lid = _INDIC_ASR.transcribe(audio_file, return_lid=True)

    return text, lid


LANGUAGE_CODES = {
    "hi": "hi", "hin": "hi", "hindi": "hi",
    "ta": "ta", "tam": "ta", "tamil": "ta",
    "bn": "bn", "ben": "bn", "bengali": "bn", "bangla": "bn",
    "te": "te", "tel": "te", "telugu": "te",
    "kn": "kn", "kan": "kn", "kannada": "kn",
    "ml": "ml", "mal": "ml", "malayalam": "ml",
    "gu": "gu", "guj": "gu", "gujarati": "gu",
    "mr": "mr", "mar": "mr", "marathi": "mr",
    "asm": "as", "as": "as", "ass": "as", "assamese": "as",
    "pa": "pa", "pan": "pa", "pu": "pa", "pun": "pa", "punjabi": "pa",
    "oriya": "or", "ori": "or", "ory": "or", "od": "or", "odi": "or", "odia": "or", "or": "or",

}


def select_recording_language(detected_languages, override=None):
    """Each recognized segment casts one vote; --lang always wins."""
    votes = Counter(x for x in detected_languages if x is not None)
    winners = sorted(k for k, n in votes.items() if n == max(votes.values())) if votes else []
    selected = normalize_language(override) if override else None
    if not override and len(winners) > 1:
        raise ValueError(f"Language vote is tied: {dict(votes)}. Set --lang explicitly.")
    if not override and not winners and detected_languages:
        raise ValueError("No recognized segment language predictions. Set --lang explicitly.")
    if not override and winners:
        selected = winners[0]
    total = sum(votes.values())
    fraction = votes.get(selected, 0) / total if total else 0.0
    return selected, {"votes": dict(votes), "valid_votes": total,
                      "total_segments": len(detected_languages),
                      "unrecognized_predictions": len(detected_languages) - total,
                      "selected_language": selected,
                      "selection_source": "override" if override else "segment_vote",
                      "selected_vote_fraction": fraction,
                      "strict_majority": fraction > 0.5, "tied_languages": winners if len(winners) > 1 else []}


def normalize_language(lid: Any) -> str:
    """Convert the LID output (e.g. 'hi', 'hin', 'Hindi', 'hi-IN') to ctc_score codes."""

    if isinstance(lid, dict):
        lid = lid.get("language") or lid.get("lang") or lid.get("label")

    if isinstance(lid, (list, tuple)):
        lid = lid[0]

    key = str(lid).strip().lower().replace("_", "-").split("-")[0]

    if key not in LANGUAGE_CODES:
        raise ValueError(
            f"Language '{lid}' is not supported by ctc_score.py. "
            f"Use --lang with one of: {sorted(set(LANGUAGE_CODES.values()))}"
        )

    return LANGUAGE_CODES[key]


# ============================================================
# 2. Word-level timestamps (ctc_score.py) mapped onto the transcript
# ============================================================

def _norm_word(word: str) -> str:
    word = unicodedata.normalize("NFC", word).lower()
    return "".join(
        ch for ch in word
        if not unicodedata.category(ch).startswith("P")
    )


def tokenize_with_offsets(text: str) -> List[Dict[str, Any]]:
    return [
        {"word": m.group(), "char_start": m.start(), "char_end": m.end()}
        for m in re.finditer(r"\S+", text)
    ]


def _spread(tokens, i1, i2, start, end):
    """Share [start, end] between tokens[i1:i2] in proportion to their length."""

    lengths = [max(len(_norm_word(tokens[i]["word"])), 1) for i in range(i1, i2)]
    total = sum(lengths)
    t = start

    for i, n in zip(range(i1, i2), lengths):
        d = (end - start) * n / total
        tokens[i]["start"], tokens[i]["end"] = t, t + d
        t += d


def align_transcript_to_ctc(
    transcript: str,
    ctc_words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Give every word of the asr_li transcript a start/end time, using the
    word timestamps produced by ctc_score.py.

    The two ASR systems may spell words differently, so words are matched
    with a sequence alignment; unmatched words get proportional /
    interpolated times from their neighbours.
    """

    tokens = tokenize_with_offsets(transcript)

    if not tokens:
        return tokens

    if not ctc_words:
        raise RuntimeError("ctc_score.py returned no word timestamps.")

    matcher = SequenceMatcher(
        None,
        [_norm_word(t["word"]) for t in tokens],
        [_norm_word(w["word"]) for w in ctc_words],
        autojunk=False,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            for k in range(i2 - i1):
                tokens[i1 + k]["start"] = ctc_words[j1 + k]["start"]
                tokens[i1 + k]["end"] = ctc_words[j1 + k]["end"]

        elif tag == "replace":
            _spread(tokens, i1, i2, ctc_words[j1]["start"], ctc_words[j2 - 1]["end"])

    # Words with no CTC counterpart: fill the gap between neighbours
    i = 0
    while i < len(tokens):

        if "start" in tokens[i]:
            i += 1
            continue

        j = i
        while j < len(tokens) and "start" not in tokens[j]:
            j += 1

        gap_start = tokens[i - 1]["end"] if i > 0 else ctc_words[0]["start"]
        gap_end = tokens[j]["start"] if j < len(tokens) else ctc_words[-1]["end"]

        _spread(tokens, i, j, gap_start, max(gap_end, gap_start))
        i = j

    return tokens


# ============================================================
# 3 + 4. NER (qwen_ner.py) and cross-check (ned_1.py)
# ============================================================

def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def cross_check(
    transcript: str,
    qwen_entities: List[Dict[str, Any]],
    ned_module: ModuleType,
) -> List[Dict[str, Any]]:
    """
    Cross-check Qwen's entities with ned_1.py.

    - Qwen EMAIL overlapping a ned_1 hit  -> cross_checked=True,
      span widened to cover both.
    - Qwen EMAIL not found by ned_1       -> kept, cross_checked=False.
    - ned_1 hit missed by Qwen            -> added (source="ned_1").
    - Other types (ned_1 only covers e-mail) -> cross_checked=None.
    """

    ned_hits = ned_module.detect_spoken_email(transcript)

    entities = []

    for entity in qwen_entities:
        entity = dict(entity, source="qwen", cross_checked=None)

        if entity["type"] == "EMAIL":
            matches = [h for h in ned_hits if _overlaps(entity, h)]
            entity["cross_checked"] = bool(matches)

            for h in matches:
                entity["start"] = min(entity["start"], h["start"])
                entity["end"] = max(entity["end"], h["end"])

            entity["text"] = transcript[entity["start"]:entity["end"]]

        entities.append(entity)

    for hit in ned_hits:
        if not any(_overlaps(hit, e) for e in entities if e["type"] == "EMAIL"):
            entities.append({
                "type": "EMAIL",
                "text": hit["text"],
                "start": hit["start"],
                "end": hit["end"],
                "source": "ned_1",
                "cross_checked": True,
            })

    entities.sort(key=lambda e: e["start"])

    return entities


# ============================================================
# 5. PII words -> audio timestamps
# ============================================================

def entity_time_spans(
    entities: List[Dict[str, Any]],
    timed_tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    for entity in entities:
        words = [
            t for t in timed_tokens
            if t["char_start"] < entity["end"] and entity["start"] < t["char_end"]
        ]

        if words:
            entity["words"] = [w["word"] for w in words]
            entity["audio_start"] = min(w["start"] for w in words)
            entity["audio_end"] = max(w["end"] for w in words)

    return entities


def merge_intervals(
    spans: List[Tuple[float, float]],
    pad: float,
) -> List[Tuple[float, float]]:

    padded = sorted((max(0.0, s - pad), e + pad) for s, e in spans)
    merged: List[List[float]] = []

    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


def seconds_to_timestr(seconds: float) -> str:
    """Format for audio_beep.time_to_ms, e.g. 75.25 -> '1:15.250'."""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:06.3f}"


def output_path_for(audio_file: str) -> str:
    stem, ext = os.path.splitext(audio_file)
    return f"{stem}_{ext}"


# ============================================================
# 0. Sample-rate check / conversion to 16 kHz
# ============================================================

TARGET_SAMPLE_RATE = 16000


def convert_mp3_to_wav(audio_file: str) -> str:
    """Decode MP3 to a WAV at its original rate/channels before resampling.

    Requires FFmpeg. A unique name avoids overwriting an existing WAV.
    """
    if os.path.splitext(audio_file)[1].lower() != ".mp3":
        return audio_file
    from pydub import AudioSegment
    source = os.path.abspath(audio_file)
    audio = AudioSegment.from_file(source, format="mp3")
    fd, converted = tempfile.mkstemp(prefix=os.path.splitext(os.path.basename(source))[0] + "_decoded_",
                                      suffix=".wav", dir=os.path.dirname(source))
    os.close(fd)
    try:
        audio.export(converted, format="wav")
    except Exception:
        os.unlink(converted)
        raise
    return converted


def ensure_16khz(audio_file: str) -> Tuple[str, int]:
    """
    Check the sample rate of the audio. If it is not 16 kHz (or not a
    mono WAV, which the ASR models also expect), write a 16 kHz mono
    WAV next to it as <name>_16k.wav and return that path.

    Returns (path_to_use, original_sample_rate).
    """

    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_file)
    original_rate = audio.frame_rate

    is_wav = audio_file.lower().endswith(".wav")

    if original_rate == TARGET_SAMPLE_RATE and audio.channels == 1 and is_wav:
        return audio_file, original_rate

    stem, _ = os.path.splitext(audio_file)
    converted = f"{stem}_16k.wav"

    (
        audio
        .set_frame_rate(TARGET_SAMPLE_RATE)
        .set_channels(1)
        .export(converted, format="wav")
    )

    return converted, original_rate


def detect_speech_segments(audio_file: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Silero settings match silero_vad_audio.py; return exact full-file offsets.

    Install: pip install silero-vad torch soundfile
    This function expects the mono 16 kHz WAV returned by ensure_16khz.
    """
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    if not 0 < threshold < 1:
        raise ValueError("vad_threshold must be between 0 and 1")
    audio, sr = sf.read(audio_file, dtype="float32")
    if sr != TARGET_SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("VAD input must be mono 16 kHz audio")
    if len(audio) == 0:
        return []
    model = load_silero_vad()
    with torch.inference_mode():
        spans = get_speech_timestamps(
            torch.from_numpy(audio), model, sampling_rate=sr,
            threshold=threshold, min_speech_duration_ms=250,
            min_silence_duration_ms=300, speech_pad_ms=100,
            max_speech_duration_s=30, return_seconds=False,
        )
    return [
        {"start_sample": int(x["start"]), "end_sample": int(x["end"]),
         "start": int(x["start"]) / sr, "end": int(x["end"]) / sr}
        for x in spans
    ]


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(
    audio_file: str,
    beep_file: str = os.path.join(HERE, "beep.wav"),
    language: Optional[str] = None,
    pad: float = 0.1,
    vad_threshold: float = 0.5,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:

    log_root = os.path.abspath(log_dir or os.path.splitext(audio_file)[0] + "_logs")
    os.makedirs(log_root, exist_ok=True)
    diagnostics = tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"), dir=log_root)
    logger = logging.getLogger("pii_pipeline." + os.path.basename(diagnostics))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(os.path.join(diagnostics, "pipeline.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    print(f"Diagnostics: {diagnostics}")
    try:
        return _run_pipeline_logged(audio_file, beep_file, language, pad, vad_threshold, diagnostics, logger)
    except Exception:
        logger.exception("Pipeline failed")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def _run_pipeline_logged(audio_file, beep_file, language, pad, vad_threshold, diagnostics, logger):
    logger.info("Input=%s language_override=%s pad=%s vad_threshold=%s", os.path.abspath(audio_file), language, pad, vad_threshold)
    if pad < 0:
        raise ValueError("pad must be nonnegative")
    if language is not None:
        language = normalize_language(language)
    # 0. Sample rate
    print("\n[0/7] Checking sample rate")
    original_audio = audio_file
    print("  Converting MP3 to WAV first (if needed)")
    wav_input = convert_mp3_to_wav(original_audio)
    if wav_input != original_audio:
        logger.info("MP3 decoded to WAV: %s", wav_input)
        print(f"  MP3 -> WAV: {wav_input}")
    audio_file, original_rate = ensure_16khz(wav_input)

    if audio_file == original_audio:
        print(f"  {original_rate} Hz mono WAV - no conversion needed")
    else:
        print(f"  {original_rate} Hz -> {TARGET_SAMPLE_RATE} Hz mono: {audio_file}")

    logger.info("Sample rate: %s -> %s; processed_audio=%s", original_rate, TARGET_SAMPLE_RATE, audio_file)
    # 1. VAD runs only after the full audio has been converted to 16 kHz.
    print("\n[1/7] Silero VAD speech segmentation")
    segments = detect_speech_segments(audio_file, vad_threshold)
    print(f"  Found {len(segments)} speech segments")
    logger.info("VAD found %s segments", len(segments))
    with open(os.path.join(diagnostics, "vad_segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    # Keep the full timeline intact. Segment WAVs are temporary model inputs;
    # only the full audio is passed to audio_beep below.
    from pydub import AudioSegment
    full_audio = AudioSegment.from_file(audio_file)
    duration = len(full_audio) / 1000.0
    transcripts, timed_tokens, segment_reports = [], [], []
    failed_segments = []
    char_offset = 0
    detected_languages = []
    if segments:
        import ctc_score
    with tempfile.TemporaryDirectory(prefix="pii_vad_") as work_dir:
        for index, segment in enumerate(segments, start=1):
            print(f"\n[2/7] ASR + language identification for segment {index}/{len(segments)}")
            segment_file = os.path.join(work_dir, f"speech_{index:04d}.wav")
            # Slice at exact sample boundaries; avoid millisecond rounding.
            chunk = full_audio.get_sample_slice(segment["start_sample"], segment["end_sample"])
            chunk.export(segment_file, format="wav")
            diagnostic = dict(segment, segment_index=index,
                              duration_seconds=segment["end"] - segment["start"],
                              sample_rate=TARGET_SAMPLE_RATE, status="processing")
            diagnostic_path = os.path.join(diagnostics, f"segment_{index:04d}.json")
            stage = "asr"
            try:
                text, lid = identify_and_transcribe(segment_file)
                diagnostic.update(transcript=text, lid=str(lid))
                stage = "language_normalization"
                try:
                    segment_language = normalize_language(lid)
                except (ValueError, IndexError, TypeError) as exc:
                    segment_language = None
                    diagnostic["language_prediction_error"] = str(exc)
                    logger.warning("Segment %s: unrecognized LID=%r", index, lid)
                diagnostic["language"] = segment_language
                detected_languages.append(segment_language)
                report = dict(segment, lid=str(lid), language=segment_language, transcript=text)
                segment_reports.append(report)
                logger.info("Segment %s: %.3f-%.3fs, duration=%.3fs, lid=%s, language=%s, transcript=%r",
                            index, segment["start"], segment["end"], diagnostic["duration_seconds"],
                            lid, segment_language, text)
                diagnostic["status"] = "asr_complete" if text.strip() else "empty_asr_transcript"
            except Exception as exc:
                failed_audio = os.path.join(diagnostics, f"failed_segment_{index:04d}.wav")
                shutil.copy2(segment_file, failed_audio)
                diagnostic.update(status="failed", failed_stage=stage,
                                  error_type=type(exc).__name__, error=str(exc),
                                  traceback=traceback.format_exc(), saved_audio=failed_audio)
                logger.exception("Segment %s failed during %s; saved audio: %s", index, stage, failed_audio)
                print(f"Failed segment saved: {failed_audio}")
                if isinstance(exc, (ImportError, FileNotFoundError, PermissionError)):
                    raise
                diagnostic["status"] = "failed_segment_redacted"
                failed_segments.append(dict(segment, segment_index=index, failed_stage=stage,
                                            error=str(exc), saved_audio=failed_audio))
                detected_languages.append(None)
                segment_reports.append(dict(segment, lid=None, language=None, transcript="",
                                            status="failed_segment_redacted"))
                continue
            finally:
                with open(diagnostic_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostic, f, ensure_ascii=False, indent=2, default=str)
        # All segment LIDs are known before any CTC call.
        try:
            result_language, language_selection = select_recording_language(detected_languages, language)
        except ValueError as exc:
            with open(os.path.join(diagnostics, "language_votes.json"), "w", encoding="utf-8") as f:
                json.dump({"votes": dict(Counter(x for x in detected_languages if x)),
                           "error": str(exc)}, f, indent=2)
            raise
        with open(os.path.join(diagnostics, "language_votes.json"), "w", encoding="utf-8") as f:
            json.dump(language_selection, f, indent=2)
        logger.info("Recording language selection: %s", language_selection)
        print(f"  Main language: {result_language}; votes: {language_selection['votes']}; source: {language_selection['selection_source']}")
        for index, (segment, report) in enumerate(zip(segments, segment_reports), start=1):
            segment_file = os.path.join(work_dir, f"speech_{index:04d}.wav")
            diagnostic_path = os.path.join(diagnostics, f"segment_{index:04d}.json")
            with open(diagnostic_path, encoding="utf-8") as f:
                diagnostic = json.load(f)
            diagnostic["ctc_language"] = result_language
            report["ctc_language"] = result_language
            if report.get("status") == "failed_segment_redacted":
                continue
            text = report["transcript"]
            if not text.strip():
                diagnostic["status"] = "empty_asr_transcript"
                with open(diagnostic_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostic, f, ensure_ascii=False, indent=2)
                continue
            print(f"[2/7] CTC segment {index}/{len(segments)} using {result_language}")
            stage = "ctc"
            try:
                ctc_result = ctc_score.transcribe_with_timestamps(segment_file, result_language)
                diagnostic["ctc_result"] = ctc_result
                stage = "timestamp_alignment"
                local_tokens = align_transcript_to_ctc(text, ctc_result["words"])
                diagnostic.update(status="success", ctc_word_count=len(ctc_result["words"]))
                logger.info("Segment %s: CTC language=%s, words=%s", index, result_language, len(ctc_result["words"]))
            except Exception as exc:
                failed_audio = os.path.join(diagnostics, f"failed_segment_{index:04d}.wav")
                shutil.copy2(segment_file, failed_audio)
                diagnostic.update(status="failed", failed_stage=stage, error_type=type(exc).__name__,
                                  error=str(exc), traceback=traceback.format_exc(), saved_audio=failed_audio)
                logger.exception("Segment %s failed during %s; saved audio: %s", index, stage, failed_audio)
                print(f"Failed segment saved: {failed_audio}")
                if isinstance(exc, (ImportError, FileNotFoundError, PermissionError)):
                    raise
                diagnostic["status"] = "failed_segment_redacted"
                report["status"] = "failed_segment_redacted"
                failed_segments.append(dict(segment, segment_index=index, failed_stage=stage,
                                            error=str(exc), saved_audio=failed_audio))
                # Retain the ASR text and map all its words conservatively to
                # this whole segment. No approximate word timings are claimed.
                local_tokens = tokenize_with_offsets(text)
                for token in local_tokens:
                    token.update(start=0.0, end=segment["end"] - segment["start"],
                                 timestamp_source="failed_segment_bounds")
            finally:
                with open(diagnostic_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostic, f, ensure_ascii=False, indent=2, default=str)
            local_duration = segment["end"] - segment["start"]
            for token in local_tokens:
                # CTC timestamps refer to the chunk; convert to the full file.
                local_start = max(0.0, min(float(token["start"]), local_duration))
                local_end = max(local_start, min(float(token["end"]), local_duration))
                token["start"] = segment["start"] + local_start
                token["end"] = segment["start"] + local_end
                token["char_start"] += char_offset
                token["char_end"] += char_offset
                token["segment_index"] = index
            timed_tokens.extend(local_tokens)
            transcripts.append(text)
            char_offset += len(text) + 1  # One space between segment transcripts.
    transcript = " ".join(transcripts)
    # 3. NER
    print("\n[3/7] PII detection (qwen_ner)")
    import qwen_ner
    qwen_entities = qwen_ner.detect_pii(transcript) if transcript else []
    print(f"  Qwen found {len(qwen_entities)} entities")

    # 4. Cross-check
    print("\n[4/7] Cross-check (ned_1)")
    ned_1 = load_definitions("ned_1.py")
    entities = cross_check(transcript, qwen_entities, ned_1) if transcript else []

    # 5. Timestamps of PII
    print("\n[5/7] Mapping PII to audio timestamps")
    entities = entity_time_spans(entities, timed_tokens)

    for e in entities:
        if "audio_start" in e:
            print(
                f"  {e['type']:<13} {e['audio_start']:7.2f}s - {e['audio_end']:7.2f}s  "
                f"[{e['source']}, cross_checked={e['cross_checked']}]  {e['text']}"
            )
        else:
            print(f"  {e['type']:<13} (no timestamp found)  {e['text']}")

    intervals = merge_intervals(
        [(e["audio_start"], e["audio_end"]) for e in entities if "audio_start" in e],
        pad=pad,
    )
    intervals = merge_intervals(intervals + [(x["start"], x["end"]) for x in failed_segments], pad=0.0)

    intervals = [(s, min(e, duration)) for s, e in intervals if s < duration]

    # 6. Beep
    # Output keeps the ORIGINAL name + "_" (beeped audio is the 16 kHz version)
    output_audio = output_path_for(original_audio)
    print(f"\n[6/7] Beeping {len(intervals)} interval(s) (audio_beep) -> {output_audio}")

    if not intervals:
        full_audio.export(output_audio, format=os.path.splitext(output_audio)[1].lstrip("."))
    else:
        audio_beep = load_definitions("audio_beep.py")
        audio_beep.replace_with_beep(
            input_audio=audio_file,
            beep_audio=beep_file,
            output_audio=output_audio,
            intervals=[(seconds_to_timestr(s), seconds_to_timestr(e)) for s, e in intervals],
        )

    logger.info("Pipeline complete: output=%s beep_intervals=%s", output_audio, intervals)
    return {
        "failed_segments": failed_segments,
        "completed_with_segment_errors": bool(failed_segments),
        "diagnostics_dir": diagnostics,
        "audio": original_audio,
        "original_sample_rate": original_rate,
        "wav_input": wav_input,
        "processed_audio": audio_file,
        "output_audio": output_audio,
        "lid": [s["lid"] for s in segment_reports],
        "language": result_language,
        "language_selection": language_selection,
        "vad_segments": segment_reports,
        "timed_tokens": timed_tokens,
        "transcript": transcript,
        "redacted_transcript": qwen_ner.redact_transcript(transcript, entities) if transcript else "",
        "entities": entities,
        "beep_intervals": intervals,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect PII in speech and beep it out.")
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--beep", default=os.path.join(HERE, "beep.wav"), help="beep sound file")
    parser.add_argument("--lang", default=None, help="main CTC language; overrides segment voting (hi, ta, bn, te, kn, ml, gu, mr, pa, as, or)")
    parser.add_argument("--pad", type=float, default=0.1, help="seconds of padding around each beep")
    parser.add_argument("--json", default=None, help="optional path to save a JSON report")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero speech threshold (0 to 1)")
    parser.add_argument("--log-dir", default=None, help="persistent diagnostic folder (default: <audio_stem>_logs)")
    args = parser.parse_args()

    result = run_pipeline(args.audio, args.beep, args.lang, args.pad, args.vad_threshold, args.log_dir)

    print("\nREDACTED TRANSCRIPT")
    print(result["redacted_transcript"])
    print(f"\nSaved: {result['output_audio']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.json}")


if __name__ == "__main__":
    main()
