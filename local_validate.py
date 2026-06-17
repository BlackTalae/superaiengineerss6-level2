import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval_normalizer import normalize_eval


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen"
OCR_FILES = [
    ("paddle", ROOT / "ocr_paddleocr_all.json"),
    ("easy", ROOT / "ocr_easyocr_all.json"),
    ("tess", ROOT / "ocr_tha_eng.json"),
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean_text(text):
    text = (text or "").translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_ocr():
    out = {}
    for name, path in OCR_FILES:
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def row_text(row, ocr_sources):
    rid = str(row["id"])
    parts = [f"bucket_{int(rid)//100}", f"idmod_{int(rid)%10}"]
    for name in ("paddle", "easy", "tess"):
        if name in ocr_sources:
            text = clean_text(ocr_sources[name].get(rid, ""))
            if text:
                parts.append(text)
    return " ".join(parts)


def score_predictions(name, preds, truths):
    norm_preds = [normalize_eval(p) for p in preds]
    norm_truths = [normalize_eval(t) for t in truths]
    correct = sum(p == t for p, t in zip(norm_preds, norm_truths))
    print(f"{name}: {correct}/{len(truths)} = {correct / max(1, len(truths)):.4f}")
    return correct


def score_validation_csv(path):
    rows = read_csv(path)
    if not {"pred", "truth"}.issubset(rows[0].keys()):
        raise ValueError(f"{path} must contain pred,truth")
    score_predictions(path.name, [r["pred"] for r in rows], [r["truth"] for r in rows])


def loo_constants(train):
    truths = [r["answer"] for r in train]
    counts = Counter(truths)
    print("Top constants, normalized scoring:")
    for ans, _ in counts.most_common(15):
        score_predictions(f"constant:{ans}", [ans] * len(truths), truths)


def loo_bucket_majority(train):
    truths = [r["answer"] for r in train]
    preds = []
    for i, row in enumerate(train):
        rest = train[:i] + train[i + 1 :]
        bucket = int(row["id"]) // 100
        bucket_answers = [r["answer"] for r in rest if int(r["id"]) // 100 == bucket]
        pool = bucket_answers or [r["answer"] for r in rest]
        preds.append(Counter(pool).most_common(1)[0][0])
    score_predictions("loo_bucket_majority", preds, truths)


def loo_nearest(train, ocr_sources):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [row_text(r, ocr_sources) for r in train]
    truths = [r["answer"] for r in train]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix)

    best = None
    for k in [1, 3, 5, 7, 9]:
        for threshold in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
            preds = []
            for i in range(len(train)):
                row_sims = sims[i].copy()
                row_sims[i] = -1
                order = row_sims.argsort()[::-1][:k]
                if row_sims[order[0]] < threshold:
                    preds.append("2")
                    continue
                votes = defaultdict(float)
                for idx in order:
                    votes[truths[idx]] += max(0.0, float(row_sims[idx]))
                preds.append(max(votes.items(), key=lambda kv: kv[1])[0])
            correct = sum(normalize_eval(p) == normalize_eval(t) for p, t in zip(preds, truths))
            key = (correct, k, -threshold)
            if best is None or key > best[0]:
                best = (key, f"loo_nearest:k{k}:t{threshold}", preds)
    score_predictions(best[1], best[2], truths)


def inspect_train():
    train = read_csv(DATA / "train.csv")
    print(f"train rows={len(train)} unique_answers={len(set(r['answer'] for r in train))}")
    print("Raw answer top:")
    for ans, count in Counter(r["answer"] for r in train).most_common(20):
        print(f"{ans!r}: {count}")
    print("Normalized answer top:")
    for ans, count in Counter(normalize_eval(r["answer"]) for r in train).most_common(20):
        print(f"{ans!r}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csv", action="append", default=[])
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    train = read_csv(DATA / "train.csv")
    if args.inspect:
        inspect_train()
    for path in args.validation_csv:
        score_validation_csv(ROOT / path)
    loo_constants(train)
    loo_bucket_majority(train)
    loo_nearest(train, load_ocr())


if __name__ == "__main__":
    main()
