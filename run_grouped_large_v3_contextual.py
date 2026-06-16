import csv
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGES_DIR = ROOT / ".asr-packages"
sys.path.insert(0, str(PACKAGES_DIR))

for dll_dir in (
    PACKAGES_DIR / "nvidia" / "cublas" / "bin",
    PACKAGES_DIR / "nvidia" / "cudnn" / "bin",
):
    os.add_dll_directory(str(dll_dir))

from faster_whisper import WhisperModel


DATA_DIR = ROOT / "individual-test-thai-call-center-asr"
AUDIO_DIR = DATA_DIR / "audio_final" / "audio"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = ROOT / "outputs"
FALLBACK_PATH = OUTPUT_DIR / "submission_large_v3_grouped_medoid.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "large_v3_contextual.checkpoint.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission_large_v3_grouped_contextual.csv"
LOG_PATH = OUTPUT_DIR / "large_v3_contextual.log"
SUFFIXES = ("_phone", "_noise", "_fast", "_slow", "_pitch")


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def base_key(file_name):
    path = Path(file_name)
    stem = path.stem
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + path.suffix


def clean_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def transcribe(model, audio_path):
    segments, _ = model.transcribe(
        str(audio_path),
        language="th",
        task="transcribe",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=True,
        vad_filter=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    return clean_text("".join(segment.text for segment in segments))


def write_csv(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    with SUBMISSION_PATH.open(newline="", encoding="utf-8-sig") as submission_file:
        expected_rows = list(csv.DictReader(submission_file))
    with FALLBACK_PATH.open(newline="", encoding="utf-8-sig") as fallback_file:
        fallback = {
            base_key(row["file_name"]): row["text"].strip()
            for row in csv.DictReader(fallback_file)
        }

    groups = defaultdict(list)
    for row in expected_rows:
        groups[base_key(row["file_name"])].append(row["file_name"])
    representatives = {
        key: key if key in file_names else file_names[0]
        for key, file_names in groups.items()
    }

    completed = {}
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open(newline="", encoding="utf-8-sig") as checkpoint_file:
            completed = {
                row["file_name"]: row["text"].strip()
                for row in csv.DictReader(checkpoint_file)
                if row["text"].strip()
            }

    log(f"Starting contextual decoding: groups={len(groups)}, resumed={len(completed)}")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    started_at = time.time()

    for index, (key, representative) in enumerate(representatives.items(), start=1):
        if key in completed:
            continue
        text = transcribe(model, AUDIO_DIR / representative)
        if not text:
            text = fallback[key]
            log(f"FALLBACK medoid transcript for {representative}")
        completed[key] = text

        if index % 20 == 0:
            write_csv(
                CHECKPOINT_PATH,
                [
                    {"file_name": group_key, "text": completed[group_key]}
                    for group_key in representatives
                    if group_key in completed
                ],
            )
            log(
                f"Progress {len(completed)}/{len(groups)} "
                f"elapsed={time.time() - started_at:.0f}s"
            )

    result = [
        {"file_name": row["file_name"], "text": completed[base_key(row["file_name"])]}
        for row in expected_rows
    ]
    write_csv(OUTPUT_PATH, result)
    write_csv(
        CHECKPOINT_PATH,
        [{"file_name": key, "text": completed[key]} for key in representatives],
    )
    log(f"Completed: rows={len(result)}, output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
