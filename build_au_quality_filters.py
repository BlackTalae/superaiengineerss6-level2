import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual_medoid.csv"
CONTEXTUAL_PATH = ROOT / "outputs" / "submission_large_v3_contextual_au_only.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_large_v3_au_quality_ratio_09.csv"


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def has_repetition(text):
    return bool(re.search(r"(.{8,}?)\1{2,}", text))


def choose_text(base, contextual):
    if not base["file_name"].startswith("AU_"):
        return contextual["text"]

    contextual_text = contextual["text"]
    base_text = base["text"]
    contextual_too_short = len(contextual_text) < len(base_text) * 0.90
    contextual_repeats = has_repetition(contextual_text) and not has_repetition(base_text)

    return base_text if contextual_too_short or contextual_repeats else contextual_text


def main():
    base_rows = read_rows(BASE_PATH)
    contextual_rows = read_rows(CONTEXTUAL_PATH)

    if len(base_rows) != len(contextual_rows):
        raise ValueError("Input submissions have different row counts")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        for base, contextual in zip(base_rows, contextual_rows):
            if base["file_name"] != contextual["file_name"]:
                raise ValueError(f"File order mismatch: {base['file_name']}")
            writer.writerow(
                {"file_name": base["file_name"], "text": choose_text(base, contextual)}
            )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
