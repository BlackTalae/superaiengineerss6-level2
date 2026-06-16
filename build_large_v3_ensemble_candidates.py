import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "large_v3_all_variants_with_confidence.csv"
OUTPUT_DIR = ROOT / "outputs"
SUFFIXES = ("_phone", "_noise", "_fast", "_slow", "_pitch")
STABLE_VARIANTS = {"original", "phone", "noise", "slow"}


def base_key(file_name):
    path = Path(file_name)
    stem = path.stem
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + path.suffix


def variant_name(file_name):
    stem = Path(file_name).stem
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            return suffix[1:]
    return "original"


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


def choose_medoid(rows):
    scores = []
    for row in rows:
        normalized_distance = sum(
            distance(row["text"], other["text"])
            / max(len(row["text"]), len(other["text"]), 1)
            for other in rows
        )
        scores.append((normalized_distance, -row["score"], row))
    return min(scores, key=lambda item: (item[0], item[1]))[2]


def write_submission(name, source_rows, selected=None):
    path = OUTPUT_DIR / name
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file_name", "text"])
        writer.writeheader()
        for row in source_rows:
            chosen = selected[base_key(row["file_name"])] if selected else row
            writer.writerow({"file_name": row["file_name"], "text": chosen["text"]})
    return path


def main():
    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        row["score"] = float(row["avg_logprob"])

    groups = defaultdict(list)
    for row in rows:
        groups[base_key(row["file_name"])].append(row)

    confidence = {}
    stable_confidence = {}
    medoid = {}
    conservative = {threshold: {} for threshold in (0.05, 0.10, 0.20)}
    selected_variants = {
        "confidence": Counter(),
        "stable_confidence": Counter(),
        "medoid": Counter(),
    }

    for key, group in groups.items():
        confidence[key] = max(group, key=lambda row: row["score"])
        stable = [
            row for row in group if variant_name(row["file_name"]) in STABLE_VARIANTS
        ]
        stable_confidence[key] = max(stable, key=lambda row: row["score"])
        medoid[key] = choose_medoid(group)
        original = next(
            (row for row in group if variant_name(row["file_name"]) == "original"),
            group[0],
        )
        for threshold in conservative:
            conservative[threshold][key] = (
                stable_confidence[key]
                if stable_confidence[key]["score"] >= original["score"] + threshold
                else original
            )
        for selection_name, selection in (
            ("confidence", confidence),
            ("stable_confidence", stable_confidence),
            ("medoid", medoid),
        ):
            selected_variants[selection_name][
                variant_name(selection[key]["file_name"])
            ] += 1

    paths = [
        write_submission("submission_large_v3_direct_variants.csv", rows),
        write_submission("submission_large_v3_grouped_confidence.csv", rows, confidence),
        write_submission(
            "submission_large_v3_grouped_stable_confidence.csv",
            rows,
            stable_confidence,
        ),
        write_submission("submission_large_v3_grouped_medoid.csv", rows, medoid),
    ]
    for threshold, selection in conservative.items():
        suffix = str(threshold).replace(".", "")
        paths.append(
            write_submission(
                f"submission_large_v3_grouped_confidence_margin_{suffix}.csv",
                rows,
                selection,
            )
        )
    for path in paths:
        print(path)
    for name, counts in selected_variants.items():
        print(name, dict(counts))
    for threshold, selection in conservative.items():
        changed = sum(
            variant_name(row["file_name"]) != "original" for row in selection.values()
        )
        print(f"confidence_margin_{threshold}: changed_groups={changed}")


if __name__ == "__main__":
    main()
