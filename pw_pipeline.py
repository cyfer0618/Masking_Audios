"""
PII audio redaction pipeline.

    audio
      -> resample to 16 kHz mono WAV if needed (<name>_16k.wav)
      -> asr_li.py     : language identification + transcription
      -> ctc_score.py   : word-level timestamps
      -> qwen_ner.py    : NAME / EMAIL / PHONE_NUMBER / PAN / AADHAAR / ADDRESS
      -> ned_1.py       : cross-check (spoken e-mail detection)
      -> map PII words to timestamps
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
        sys.path.insert(0, model_dir)

        from indic_transcribe import IndicTranscribe
        _INDIC_ASR = IndicTranscribe.from_pretrained(model_dir)

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
}


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


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(
    audio_file: str,
    beep_file: str = os.path.join(HERE, "beep.wav"),
    language: Optional[str] = None,
    pad: float = 0.1,
) -> Dict[str, Any]:

    # 0. Sample rate
    print("\n[0/6] Checking sample rate")
    original_audio = audio_file
    audio_file, original_rate = ensure_16khz(original_audio)

    if audio_file == original_audio:
        print(f"  {original_rate} Hz mono WAV - no conversion needed")
    else:
        print(f"  {original_rate} Hz -> {TARGET_SAMPLE_RATE} Hz mono: {audio_file}")

    # 1. LID + transcription
    print("\n[1/6] Language identification + transcription (asr_li)")
    transcript, lid = identify_and_transcribe(audio_file)
    language = normalize_language(language or lid)
    print(f"  LID        : {lid} -> {language}")
    print(f"  Transcript : {transcript}")

    # 2. Word timestamps
    print("\n[2/6] Word-level timestamps (ctc_score)")
    import ctc_score
    ctc_result = ctc_score.transcribe_with_timestamps(audio_file, language)
    timed_tokens = align_transcript_to_ctc(transcript, ctc_result["words"])
    print(f"  {len(ctc_result['words'])} CTC words, {len(timed_tokens)} transcript words timed")

    # 3. NER
    print("\n[3/6] PII detection (qwen_ner)")
    import qwen_ner
    qwen_entities = qwen_ner.detect_pii(transcript)
    print(f"  Qwen found {len(qwen_entities)} entities")

    # 4. Cross-check
    print("\n[4/6] Cross-check (ned_1)")
    ned_1 = load_definitions("ned_1.py")
    entities = cross_check(transcript, qwen_entities, ned_1)

    # 5. Timestamps of PII
    print("\n[5/6] Mapping PII to audio timestamps")
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

    # 6. Beep
    # Output keeps the ORIGINAL name + "_" (beeped audio is the 16 kHz version)
    output_audio = output_path_for(original_audio)
    print(f"\n[6/6] Beeping {len(intervals)} interval(s) (audio_beep) -> {output_audio}")

    audio_beep = load_definitions("audio_beep.py")
    audio_beep.replace_with_beep(
        input_audio=audio_file,
        beep_audio=beep_file,
        output_audio=output_audio,
        intervals=[(seconds_to_timestr(s), seconds_to_timestr(e)) for s, e in intervals],
    )

    return {
        "audio": original_audio,
        "original_sample_rate": original_rate,
        "processed_audio": audio_file,
        "output_audio": output_audio,
        "lid": str(lid),
        "language": language,
        "transcript": transcript,
        "redacted_transcript": qwen_ner.redact_transcript(transcript, entities),
        "entities": entities,
        "beep_intervals": intervals,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect PII in speech and beep it out.")
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--beep", default=os.path.join(HERE, "beep.wav"), help="beep sound file")
    parser.add_argument("--lang", default=None, help="override detected language (hi, ta, bn, te, kn, ml, gu)")
    parser.add_argument("--pad", type=float, default=0.1, help="seconds of padding around each beep")
    parser.add_argument("--json", default=None, help="optional path to save a JSON report")
    args = parser.parse_args()

    result = run_pipeline(args.audio, args.beep, args.lang, args.pad)

    print("\nREDACTED TRANSCRIPT")
    print(result["redacted_transcript"])
    print(f"\nSaved: {result['output_audio']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Report: {args.json}")


if __name__ == "__main__":
    main()
