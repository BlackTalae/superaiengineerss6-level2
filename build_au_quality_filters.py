import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual_medoid.csv"
CONTEXTUAL_PATH = ROOT / "outputs" / "submission_large_v3_contextual_au_only.csv"
OUTPUT_DIR = ROOT / "outputs"


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def has_repetition(text):
    return bool(re.search(r"(.{8,}?)\1{2,}", text))


def write_candidate(name, base_rows, contextual_rows, choose):
    path = OUTPUT_DIR / name
    reverted = 0
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        for base, contextual in zip(base_rows, contextual_rows):
            text = choose(base, contextual)
            reverted += text != contextual["text"]
            writer.writerow({"file_name": base["file_name"], "text": text})
    print(f"{path}: reverted={reverted}")


def main():
    base_rows = read_rows(BASE_PATH)
    contextual_rows = read_rows(CONTEXTUAL_PATH)

    write_candidate(
        "submission_large_v3_au_contextual_no_repetition.csv",
        base_rows,
        contextual_rows,
        lambda base, contextual: (
            base["text"]
            if base["file_name"].startswith("AU_")
            and has_repetition(contextual["text"])
            and not has_repetition(base["text"])
            else contextual["text"]
        ),
    )

    for ratio in (0.95, 0.90, 0.80):
        suffix = str(ratio).replace(".", "")
        write_candidate(
            f"submission_large_v3_au_quality_ratio_{suffix}.csv",
            base_rows,
            contextual_rows,
            lambda base, contextual, ratio=ratio: (
                base["text"]
                if base["file_name"].startswith("AU_")
                and (
                    len(contextual["text"]) < len(base["text"]) * ratio
                    or (
                        has_repetition(contextual["text"])
                        and not has_repetition(base["text"])
                    )
                )
                else contextual["text"]
            ),
        )


if __name__ == "__main__":
    main()
