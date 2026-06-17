import argparse
import csv
import json
import math
import re
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


def load_ocr():
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in OCR_FILES
        if path.exists()
    }


def clean(text):
    text = (text or "").translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    text = text.replace("×", " x ").replace("÷", " / ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def all_text(rid, ocr):
    parts = []
    for name in ("paddle", "easy", "tess"):
        if name in ocr:
            txt = clean(ocr[name].get(str(rid), ""))
            if txt:
                parts.append(txt)
    return "\n".join(parts)


def ints(text):
    return [int(x.replace(",", "")) for x in re.findall(r"(?<![\d.])-?\d[\d,]*(?![\d.])", text)]


def lcm_many(values):
    ans = 1
    for value in values:
        ans = ans * value // math.gcd(ans, value)
    return ans


def divisors(n):
    out = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
    return out


def solve_page_digits(text):
    nums = ints(text)
    if "เลขโดด" not in text or "เลขหน้า" not in text or not nums:
        return None
    total_digits = max(nums)
    if total_digits <= 9:
        return None
    pages = 0
    used = 0
    digits = 1
    start = 1
    while True:
        end = 10**digits - 1
        block_count = end - start + 1
        block_digits = block_count * digits
        if used + block_digits >= total_digits:
            remain = total_digits - used
            if remain % digits:
                return None
            pages = start + remain // digits - 1
            return str(pages)
        used += block_digits
        start = end + 1
        digits += 1
        if digits > 7:
            return None


def solve_rule(text):
    compact = clean(text)
    low = compact.lower()
    nums = ints(compact)

    if ("ค.ร.น" in compact or "ครน" in compact) and len(nums) >= 2 and re.search(r"(จงหา|หา)\s*ค\.?ร\.?น", compact):
        values = [n for n in nums if 1 < n < 10000]
        # Prefer the numbers after "ของ"; this avoids choice labels.
        tail_match = re.search(r"(?:ค\.ร\.น\.?|ครน)[^0-9]{0,20}ของ(.+)", compact)
        if tail_match:
            tail_nums = ints(tail_match.group(1))
            if len(tail_nums) >= 2:
                values = [n for n in tail_nums if 1 < n < 10000][:4]
        if 2 <= len(values) <= 5:
            return str(lcm_many(values)), "lcm"

    if ("ห.ร.ม" in compact or "หรม" in compact) and len(nums) >= 2:
        values = [n for n in nums if n > 0]
        if 2 <= len(values) <= 5:
            ans = values[0]
            for n in values[1:]:
                ans = math.gcd(ans, n)
            return str(ans), "gcd"

    if "รากที่สาม" in compact and nums:
        n = max(nums, key=abs)
        root = round(abs(n) ** (1 / 3))
        if root**3 == abs(n):
            return str(root if n >= 0 else -root), "cube_root"

    if "ธนบัตร" in compact and "รวม" in compact and len(nums) >= 4:
        denom_values = [int(x) for x in re.findall(r"ฉบับละ\s*(\d+)", compact)]
        pair_match = re.search(r"ฉบับละ\s*(\d+)\s*บาท\s*และ\s*(\d+)\s*บาท", compact)
        if pair_match:
            denom_values = [int(pair_match.group(1)), int(pair_match.group(2))]
        total_match = re.search(r"เงิน\s*([\d,]+)\s*บาท", compact)
        count_match = re.search(r"รวม\s*(\d+)\s*ฉบับ", compact)
        asks_count = "มีกี่ฉบับ" in compact
        if total_match and count_match and asks_count and len(denom_values) >= 2:
            total = int(total_match.group(1).replace(",", ""))
            count = int(count_match.group(1))
            a, b = sorted(denom_values[:2], reverse=True)
            numerator = total - b * count
            denom = a - b
            if denom and numerator % denom == 0:
                num_a = numerator // denom
                if 0 <= num_a <= count:
                    return str(num_a), "banknotes"

    if "ไปจ่ายตลาด" in compact and "อย่างน้อย 2" in compact and len(nums) >= 4:
        small = [n for n in nums if 1 < n <= 30]
        horizon = max([n for n in nums if n > 30], default=0)
        if len(small) >= 3 and horizon:
            a, b, c = small[:3]
            days = set()
            for x, y in [(a, b), (a, c), (b, c)]:
                step = lcm_many([x, y])
                days.update(range(step, horizon + 1, step))
            return str(len(days)), "market_lcm"

    if "สถานีละ" in compact and "เปลี่ยนรถไฟฟ้า" in compact:
        floats = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", compact)]
        if len(floats) >= 6:
            minutes = [x for x in floats if x >= 10]
            per_station = next((x for x in floats if 0 < x < 10 and abs(x - round(x)) > 1e-9), None)
            transfers = re.search(r"เปลี่ยนรถไฟฟ้า\s*(\d+)\s*ครั้ง\s*ครั้งละ\s*(\d+)", compact)
            bus_match = re.search(r"ใช้เวลา\s*(\d+)\s*นาที", compact)
            station_match = re.search(r"อีก\s*(\d+)\s*สถานี", compact)
            walk_match = re.search(r"เดินเท้า(?:ต่อ)?อีก\s*(\d+)\s*นาที", compact)
            if bus_match and station_match and walk_match and per_station is not None and transfers:
                bus = float(bus_match.group(1))
                stations = float(station_match.group(1))
                walk = float(walk_match.group(1))
                transfer_count = float(transfers.group(1))
                transfer_min = float(transfers.group(2))
                ans = bus + stations * per_station + transfer_count * transfer_min + walk
                return str(int(ans) if ans.is_integer() else ans), "travel_time"

    if "ปากกา" in compact and "ราคาเป็นจำนวนเต็ม" in compact and len(nums) >= 2:
        amounts = [n for n in nums if n > 20]
        if len(amounts) >= 2:
            a, b = amounts[-2], amounts[-1]
            common = sorted([d for d in divisors(math.gcd(a, b)) if d > 10])
            if common:
                price = common[0]
                return str(a // price + b // price), "same_price"

    if ("ไม่มี 3 หรือ 5 เป็นตัวประกอบ" in compact or "ไม่มี3หรือ5เป็นตัวประกอบ" in compact.replace(" ", "")) and nums:
        n = max(nums)
        count = n - n // 3 - n // 5 + n // 15
        return str(count), "not_factor_3_5"

    if "ผลบวก" in compact and "จำนวนนับ" in compact and "หาร" in compact and "ลงตัว" in compact and nums:
        if "และ" in compact and len([n for n in nums if n > 10]) >= 2:
            bigs = [n for n in nums if n > 10][:2]
            n = math.gcd(bigs[0], bigs[1])
        else:
            n = max(nums)
        if n > 10:
            return str(sum(divisors(n))), "sum_divisors"

    if "จำนวนเต็มบวกที่หาร" in compact and "ลงตัว" in compact and "พหุคูณของ" in compact and len(nums) >= 2:
        n_match = re.search(r"หาร\s*(\d+)\s*ลงตัว", compact)
        k_match = re.search(r"พหุคูณของ\s*(\d+)", compact)
        if n_match and k_match:
            n = int(n_match.group(1))
            k = int(k_match.group(1))
            return str(sum(1 for d in divisors(n) if d % k == 0)), "count_divisors_multiple"

    page = solve_page_digits(compact)
    if page is not None:
        return page, "page_digits"

    return None, None


def validate():
    ocr = load_ocr()
    train = read_csv(DATA / "train.csv")
    hits = []
    for row in train:
        pred, rule = solve_rule(all_text(row["id"], ocr))
        if pred is None:
            continue
        ok = normalize_eval(pred) == normalize_eval(row["answer"])
        hits.append((row["id"], rule, pred, row["answer"], ok))
    correct = sum(1 for *_, ok in hits if ok)
    print(f"rule coverage={len(hits)}/{len(train)} correct={correct}/{len(hits) if hits else 1} train_total={correct}/{len(train)}")
    by_rule = {}
    for _, rule, _, _, ok in hits:
        total, good = by_rule.get(rule, (0, 0))
        by_rule[rule] = (total + 1, good + int(ok))
    for rule, (total, good) in sorted(by_rule.items(), key=lambda kv: (-kv[1][1], kv[0])):
        print(f"{rule}: {good}/{total}")
    for item in hits[:80]:
        print(item)


def make_submission(base_file="submission.csv", out_file="submission_rules_overlay.csv"):
    ocr = load_ocr()
    test = read_csv(DATA / "test.csv")
    base = {r["id"]: r["answer"] for r in read_csv(ROOT / base_file)}
    rows = []
    audit = []
    for row in test:
        rid = row["id"]
        pred, rule = solve_rule(all_text(rid, ocr))
        if pred is None:
            pred = base.get(rid, "2")
            rule = "base"
        rows.append({"id": rid, "answer": pred})
        audit.append({"id": rid, "answer": pred, "source": rule})
    with (ROOT / out_file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / out_file.replace(".csv", "_audit.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "source"])
        writer.writeheader()
        writer.writerows(audit)
    print(ROOT / out_file)
    print({k: sum(1 for r in audit if r["source"] == k) for k in sorted(set(r["source"] for r in audit))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--base", default="submission.csv")
    parser.add_argument("--out", default="submission_rules_overlay.csv")
    args = parser.parse_args()
    validate()
    if args.submit:
        make_submission(args.base, args.out)


if __name__ == "__main__":
    main()
