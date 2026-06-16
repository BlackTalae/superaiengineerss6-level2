import csv
import os
import sys
import time
import unicodedata
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
FALLBACK_PATH = OUTPUT_DIR / "submission_large_v3_grouped.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "large_v3_all_variants.checkpoint.csv"
RESULT_PATH = OUTPUT_DIR / "large_v3_all_variants_with_confidence.csv"
LOG_PATH = OUTPUT_DIR / "large_v3_all_variants.log"
FIELDS = ["file_name", "text", "avg_logprob"]


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def clean_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def transcribe(model, audio_path, vad_filter=True):
    segments, _ = model.transcribe(
        str(audio_path),
        language="th",
        task="transcribe",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
        without_timestamps=True,
    )
    segments = list(segments)
    text = clean_text("".join(segment.text for segment in segments))
    weights = [max(segment.end - segment.start, 0.01) for segment in segments]
    total_weight = sum(weights)
    avg_logprob = (
        sum(segment.avg_logprob * weight for segment, weight in zip(segments, weights))
        / total_weight
        if total_weight
        else -99.0
    )
    return text, avg_logprob


def write_csv(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    with SUBMISSION_PATH.open(newline="", encoding="utf-8-sig") as submission_file:
        expected_rows = list(csv.DictReader(submission_file))
    with FALLBACK_PATH.open(newline="", encoding="utf-8-sig") as fallback_file:
        fallback = {row["file_name"]: row["text"] for row in csv.DictReader(fallback_file)}

    completed = {}
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open(newline="", encoding="utf-8-sig") as checkpoint_file:
            completed = {
                row["file_name"]: row
                for row in csv.DictReader(checkpoint_file)
                if row["text"].strip()
            }

    log(f"Starting large-v3 all variants: total={len(expected_rows)}, resumed={len(completed)}")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    started_at = time.time()

    for index, row in enumerate(expected_rows, start=1):
        file_name = row["file_name"]
        if file_name in completed:
            continue

        text, score = transcribe(model, AUDIO_DIR / file_name)
        if not text:
            text, score = transcribe(model, AUDIO_DIR / file_name, vad_filter=False)
        if not text:
            text = fallback[file_name]
            score = -99.0
            log(f"FALLBACK grouped transcript for {file_name}")

        completed[file_name] = {
            "file_name": file_name,
            "text": text,
            "avg_logprob": f"{score:.8f}",
        }

        if index % 50 == 0:
            rows = [
                completed[item["file_name"]]
                for item in expected_rows
                if item["file_name"] in completed
            ]
            write_csv(CHECKPOINT_PATH, rows)
            log(
                f"Progress {len(completed)}/{len(expected_rows)} "
                f"elapsed={time.time() - started_at:.0f}s"
            )

    rows = [completed[row["file_name"]] for row in expected_rows]
    write_csv(RESULT_PATH, rows)
    write_csv(CHECKPOINT_PATH, rows)
    log(f"Completed: rows={len(rows)}, result={RESULT_PATH}")


if __name__ == "__main__":
    main()
