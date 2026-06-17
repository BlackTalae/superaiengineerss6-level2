import argparse
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"


def load_rows(split: str):
    frames = []
    if split in ("train", "all"):
        frames.append(pd.read_csv(DATA / "train.csv", dtype={"id": str}))
    if split in ("test", "all"):
        frames.append(pd.read_csv(DATA / "test.csv", dtype={"id": str}))
    rows = pd.concat(frames, ignore_index=True)
    rows["id"] = rows["id"].astype(str)
    return rows.drop_duplicates("id")


def image_path(row):
    path = DATA / str(row["image_path"])
    if path.exists():
        return path
    nested = DATA / "images" / str(row["image_path"])
    if nested.exists():
        return nested
    return DATA / "images" / "images" / f"{row['id']}.jpg"


def load_existing(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def make_easyocr():
    import easyocr

    reader = easyocr.Reader(["th", "en"], gpu=False, verbose=False)

    def run(path: Path):
        return "\n".join(reader.readtext(str(path), detail=0, paragraph=True)).strip()

    return run


def make_paddleocr():
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="th",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    def flatten(result):
        texts = []
        for item in result:
            if isinstance(item, dict):
                if "rec_texts" in item:
                    texts.extend(str(text) for text in item["rec_texts"])
                elif "text" in item:
                    texts.append(str(item["text"]))
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                        rec = sub[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            texts.append(str(rec[0]))
        return " ".join(texts).strip()

    def run(path: Path):
        return flatten(ocr.predict(str(path)))

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["easyocr", "paddleocr"], required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    out_path = ROOT / args.out
    results = load_existing(out_path)
    rows = load_rows(args.split)
    if args.limit:
        rows = rows.head(args.limit)

    run_ocr = make_easyocr() if args.engine == "easyocr" else make_paddleocr()
    total = len(rows)
    started = time.time()

    for pos, (_, row) in enumerate(rows.iterrows(), start=1):
        rid = str(row["id"])
        if rid in results and results[rid].strip():
            continue
        path = image_path(row)
        try:
            text = run_ocr(path)
            results[rid] = text
            print(f"[{pos}/{total}] {rid}: {text[:120]}")
        except Exception as exc:
            results[rid] = ""
            print(f"[{pos}/{total}] {rid}: ERROR {exc}")
        if pos % args.save_every == 0:
            save_json(out_path, results)

    save_json(out_path, results)
    elapsed = time.time() - started
    print(f"Saved {len(results)} rows to {out_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
