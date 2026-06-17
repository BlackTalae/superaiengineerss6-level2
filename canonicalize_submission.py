import argparse
import csv
from pathlib import Path

from eval_normalizer import normalize_eval


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    in_path = ROOT / args.input
    out_name = args.out or f"{in_path.stem}_normalized.csv"
    rows = []
    with in_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            answer = normalize_eval(row["answer"]) or "2"
            rows.append({"id": row["id"], "answer": answer})

    out_path = ROOT / out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
