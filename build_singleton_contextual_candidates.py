import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "individual-test-thai-call-center-asr" / "sample_submission.csv"
MEDOID_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual_medoid.csv"
CONTEXTUAL_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual.csv"
OUTPUT_DIR = ROOT / "outputs"
SUFFIXES = ("_phone", "_noise", "_fast", "_slow", "_pitch")


def base_key(file_name):
    path = Path(file_name)
    stem = path.stem
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + path.suffix


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def write_candidate(name, rows):
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main():
    sample_rows = read_rows(SAMPLE_PATH)
    medoid_rows = read_rows(MEDOID_PATH)
    contextual_rows = read_rows(CONTEXTUAL_PATH)
    counts = Counter(base_key(row["file_name"]) for row in sample_rows)

    hybrid_singleton = []
    hybrid_au = []
    hybrid_long_singleton = []

    for medoid, contextual in zip(medoid_rows, contextual_rows):
        file_name = medoid["file_name"]
        singleton = counts[base_key(file_name)] == 1
        contextual_text = contextual["text"]
        medoid_text = medoid["text"]

        hybrid_singleton.append(
            {"file_name": file_name, "text": contextual_text if singleton else medoid_text}
        )
        hybrid_au.append(
            {
                "file_name": file_name,
                "text": contextual_text if file_name.startswith("AU_") else medoid_text,
            }
        )
        hybrid_long_singleton.append(
            {
                "file_name": file_name,
                "text": (
                    contextual_text
                    if singleton and len(medoid_text) >= 200
                    else medoid_text
                ),
            }
        )

    write_candidate("submission_large_v3_contextual_singletons.csv", hybrid_singleton)
    write_candidate("submission_large_v3_contextual_au_only.csv", hybrid_au)
    write_candidate(
        "submission_large_v3_contextual_long_singletons.csv",
        hybrid_long_singleton,
    )


if __name__ == "__main__":
    main()
