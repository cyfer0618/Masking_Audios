#!/usr/bin/env python3
"""
ctc_score.py

Local-first AI4Bharat IndicConformer CTC timestamp extraction.

For the requested language:
1) reuse an existing local .nemo file when available;
2) otherwise download the .nemo artifact from Hugging Face;
3) always load the local archive with ASRModel.restore_from();
4) force the hybrid model to the CTC decoder;
5) return word-level timestamps.

Supported: hi, ta, bn, te, kn, ml, gu.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import nemo.collections.asr as nemo_asr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "nemo_models"

MODEL_CONFIG: Dict[str, Dict[str, str]] = {
    "hi": {
        "repo_id": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_hi_hybrid_rnnt_large.nemo",
    },
    "ta": {
        "repo_id": "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_ta_hybrid_rnnt_large.nemo",
    },
    "bn": {
        "repo_id": "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_bn_hybrid_rnnt_large.nemo",
    },
    "te": {
        "repo_id": "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_te_hybrid_rnnt_large.nemo",
    },
    "kn": {
        "repo_id": "ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_kn_hybrid_rnnt_large.nemo",
    },
    "ml": {
        "repo_id": "ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_ml_hybrid_rnnt_large.nemo",
    },
    "gu": {
        "repo_id": "ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large",
        "filename": "indicconformer_stt_gu_hybrid_rnnt_large.nemo",
    },
}

MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _language(language: str) -> str:
    language = str(language).lower().strip()
    if language not in MODEL_CONFIG:
        raise ValueError(
            f"Unsupported language {language!r}; supported: "
            f"{', '.join(sorted(MODEL_CONFIG))}"
        )
    return language


def _dedupe_dirs(*dirs: Optional[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        if item is None:
            continue
        path = Path(item).expanduser().resolve()
        if str(path) not in seen:
            seen.add(str(path))
            out.append(path)
    return out


def _search_dirs(model_dir: Optional[Path]) -> list[Path]:
    env_dir = os.environ.get("PW_NEMO_MODEL_DIR") or os.environ.get(
        "INDICCONFORMER_MODEL_DIR"
    )
    return _dedupe_dirs(
        Path(model_dir) if model_dir is not None else None,
        Path(env_dir) if env_dir else None,
        SCRIPT_DIR,
        Path.cwd(),
        DEFAULT_MODEL_DIR,
    )


def _target_dir(model_dir: Optional[Path]) -> Path:
    env_dir = os.environ.get("PW_NEMO_MODEL_DIR") or os.environ.get(
        "INDICCONFORMER_MODEL_DIR"
    )
    if model_dir is not None:
        target = Path(model_dir).expanduser().resolve()
    elif env_dir:
        target = Path(env_dir).expanduser().resolve()
    else:
        target = DEFAULT_MODEL_DIR.resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_local_nemo(
    language: str,
    model_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return an already-downloaded language-specific .nemo file, if present."""
    language = _language(language)
    expected = MODEL_CONFIG[language]["filename"]

    for directory in _search_dirs(model_dir):
        exact = directory / expected
        if exact.is_file():
            return exact.resolve()

    marker = f"_stt_{language}_"
    for directory in _search_dirs(model_dir):
        if not directory.is_dir():
            continue
        matches = sorted(
            p.resolve()
            for p in directory.glob("*.nemo")
            if marker in p.name.lower()
        )
        if matches:
            if len(matches) > 1:
                logging.warning(
                    "Multiple .nemo files match %s in %s; using %s",
                    language,
                    directory,
                    matches[0],
                )
            return matches[0]

    return None


def download_nemo_model(
    language: str,
    model_dir: Optional[Path] = None,
) -> Path:
    """Download the required .nemo artifact only when it is not local."""
    language = _language(language)
    config = MODEL_CONFIG[language]
    repo_id = config["repo_id"]
    expected = config["filename"]
    target = _target_dir(model_dir)

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import (
            EntryNotFoundError,
            GatedRepoError,
            HfHubHTTPError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install with: pip install -U huggingface_hub"
        ) from exc

    logging.info(
        "No local .nemo for %s. Downloading %s/%s -> %s",
        language,
        repo_id,
        expected,
        target,
    )

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=expected,
            local_dir=str(target),
        )
        return Path(downloaded).resolve()

    except EntryNotFoundError:
        # Some repositories have changed artifact names over time. If the
        # configured filename is absent, discover the unique .nemo artifact.
        try:
            files = HfApi().list_repo_files(repo_id=repo_id)
        except (GatedRepoError, HfHubHTTPError) as exc:
            raise RuntimeError(
                f"Cannot access {repo_id}. Accept the model terms and run "
                "`hf auth login`, then retry."
            ) from exc

        candidates = [name for name in files if name.lower().endswith(".nemo")]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely choose a .nemo file in {repo_id}: {candidates!r}"
            )

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=candidates[0],
                local_dir=str(target),
            )
            return Path(downloaded).resolve()
        except (GatedRepoError, HfHubHTTPError) as exc:
            raise RuntimeError(
                f"Cannot download {repo_id}. Accept the model terms and run "
                "`hf auth login`, then retry."
            ) from exc

    except (GatedRepoError, HfHubHTTPError) as exc:
        raise RuntimeError(
            f"Cannot download {repo_id}. Accept the model terms and run "
            "`hf auth login`, then retry."
        ) from exc


def resolve_nemo_model(
    language: str,
    model_dir: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """
    Return (path, downloaded_now).

    This is strictly local-first: Hugging Face is contacted only if no suitable
    .nemo archive is found locally.
    """
    language = _language(language)
    local = find_local_nemo(language, model_dir)
    if local is not None:
        logging.info("Using local .nemo model: %s", local)
        return local, False

    path = download_nemo_model(language, model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded model file is missing: {path}")
    logging.info("Downloaded .nemo model: %s", path)
    return path, True


def load_asr_model(
    language: str,
    model_dir: Optional[Path] = None,
):
    """Load the model only via ASRModel.restore_from(local_nemo_path)."""
    language = _language(language)
    nemo_path, downloaded_now = resolve_nemo_model(language, model_dir)
    key = (language, str(nemo_path))

    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    logging.info("Restoring NeMo archive: %s", nemo_path)
    try:
        model = nemo_asr.models.ASRModel.restore_from(str(nemo_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore {nemo_path}. AI4Bharat IndicConformer may "
            "require the AI4Bharat NeMo nemo-v2 branch."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()
    model = model.to(device)

    # Required for timestamp extraction from the hybrid model.
    model.cur_decoder = "ctc"

    setattr(model, "_pw_nemo_path", str(nemo_path))
    setattr(model, "_pw_downloaded_now", downloaded_now)
    setattr(model, "_pw_repo_id", MODEL_CONFIG[language]["repo_id"])

    MODEL_CACHE[key] = model
    logging.info("CTC model ready on %s", device)
    return model


def transcribe_with_timestamps(
    audio_file: str,
    language: str,
    model_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Return CTC transcript and word timestamps using the local .nemo archive."""
    language = _language(language)
    audio_path = Path(audio_file).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_asr_model(
        language,
        Path(model_dir) if model_dir is not None else None,
    )

    hypotheses = model.transcribe(
        [str(audio_path)],
        batch_size=1,
        return_hypotheses=True,
        timestamps=True,
        language_id=language,
    )
    if not hypotheses:
        raise RuntimeError("NeMo returned no hypothesis.")

    hypothesis = hypotheses[0]
    timestamp_data = getattr(hypothesis, "timestamp", None)
    if timestamp_data is None:
        raise RuntimeError(
            "NeMo returned no timestamps. Verify your AI4Bharat NeMo version "
            "supports timestamps=True for this model."
        )

    words = [
        {
            "word": str(item["word"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in timestamp_data.get("word", [])
    ]
    if not words:
        raise RuntimeError("NeMo returned an empty word-timestamp list.")

    return {
        "language": language,
        "transcript": str(hypothesis.text),
        "words": words,
        "model_path": str(getattr(model, "_pw_nemo_path")),
        "model_repo": str(getattr(model, "_pw_repo_id")),
        "model_downloaded_now": bool(getattr(model, "_pw_downloaded_now")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndicConformer CTC word timestamps using a local-first .nemo model."
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument(
        "--language", "-l", required=True, choices=sorted(MODEL_CONFIG)
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing/downloading .nemo files. "
            "Default: nemo_models beside ctc_score.py."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    result = transcribe_with_timestamps(
        str(args.audio_file),
        args.language,
        model_dir=args.model_dir,
    )

    print(f"Model: {result['model_path']}")
    print(f"Transcript: {result['transcript']}")
    print(f"{'Word':<30}{'Start':>12}{'End':>12}")
    print("-" * 54)
    for word in result["words"]:
        print(
            f"{word['word']:<30}{word['start']:>12.2f}{word['end']:>12.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
