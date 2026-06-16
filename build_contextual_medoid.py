import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VARIANTS_PATH = ROOT / "outputs" / "large_v3_all_variants_with_confidence.csv"
CONTEXTUAL_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual.csv"
OUTPUT_PATH = ROOT / "outputs" / "submission_large_v3_grouped_contextual_medoid.csv"
SUFFIXES = ("_phone", "_noise", "_fast", "_slow", "_pitch")


def base_key(file_name):
    path = Path(file_name)
    stem = path.stem
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + path.suffix


def distance(left, right):
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_char in enumerate(right, start=1):
        current = [row]
        for column, left_char in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def choose_medoid(texts):
    unique_texts = list(dict.fromkeys(texts))
    return min(
        unique_texts,
        key=lambda text: sum(
            distance(text, other) / max(len(text), len(other), 1)
            for other in texts
        ),
    )


def main():
    with VARIANTS_PATH.open(newline="", encoding="utf-8-sig") as variants_file:
        rows = list(csv.DictReader(variants_file))
    with CONTEXTUAL_PATH.open(newline="", encoding="utf-8-sig") as contextual_file:
        contextual = {
            base_key(row["file_name"]): row["text"] for row in csv.DictReader(contextual_file)
        }

    groups = defaultdict(list)
    for row in rows:
        groups[base_key(row["file_name"])].append(row["text"])

    selected = {
        key: choose_medoid(texts + [contextual[key]]) for key, texts in groups.items()
    }

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"file_name": row["file_name"], "text": selected[base_key(row["file_name"])]}
            )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
