import re
from fractions import Fraction


THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

UNITS = [
    "ตารางเซนติเมตร",
    "ลูกบาศก์เซนติเมตร",
    "ลูกบาศก์หน่วย",
    "ตารางหน่วย",
    "เซนติเมตร",
    "มิลลิเมตร",
    "กิโลเมตร",
    "เมตร",
    "องศา",
    "หน่วย",
    "จำนวน",
    "วิธี",
    "แบบ",
    "ค่า",
    "ร้อยละ",
    "ดอลลาร์",
    "บาท",
    "ปี",
    "คน",
    "วัน",
    "นาที",
    "ครั้ง",
    "ลูก",
    "ใบ",
    "รูป",
    "ข้อ",
    "degrees",
    "degree",
    "square centimeters",
    "square centimeter",
    "centimeters",
    "centimeter",
    "years old",
]


def _expand_latex(text: str) -> str:
    # Expand innermost simple fractions repeatedly.
    frac_re = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new = frac_re.sub(r"(\1)/(\2)", text)
        if new == text:
            break
        text = new

    sqrt_brace_re = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    text = sqrt_brace_re.sub(r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*([0-9a-zA-Z]+)", r"sqrt(\1)", text)

    replacements = {
        r"\pi": "pi",
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
        r"\pm": "+-",
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\;": "",
        r"\:": "",
        r"\!": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\\overrightarrow\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\overline\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\vec\s*\{([^{}]+)\}", r"\1", text)
    return text


def normalize_eval(answer) -> str:
    text = "" if answer is None else str(answer)
    text = text.strip().lower().translate(THAI_DIGITS)
    text = text.replace("$", "")
    text = _expand_latex(text)

    for unit in sorted(UNITS, key=len, reverse=True):
        text = re.sub(re.escape(unit), "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[{}\\,]", "", text)
    text = re.sub(r"^\((-?\d+)\)$", r"\1", text)
    text = re.sub(r"\((-?\d+)\)/\((-?\d+)\)", r"\1/\2", text)

    # Host examples imply sqrt(3) and 6sqrt(3) compare as sqrt3 / 6sqrt3.
    text = re.sub(r"sqrt\((-?\d+)\)", r"sqrt\1", text)

    try:
        value = Fraction(text)
        if value.denominator == 1:
            text = str(value.numerator)
    except Exception:
        try:
            value = float(text)
            if value.is_integer():
                text = str(int(value))
        except Exception:
            pass
    return text


def accuracy(preds, truths) -> float:
    pairs = list(zip(preds, truths))
    if not pairs:
        return 0.0
    return sum(normalize_eval(p) == normalize_eval(t) for p, t in pairs) / len(pairs)


if __name__ == "__main__":
    examples = [
        ("20 ตารางเซนติเมตร", "20"),
        ("30 องศา", "30"),
        ("$6\\sqrt{3}$", "6sqrt3"),
        ("$\\frac{17}{10}$", "17/10"),
        ("๒๕", "25"),
        ("2.0", "2"),
    ]
    for raw, expected in examples:
        got = normalize_eval(raw)
        print(raw, got, "OK" if got == expected else f"expected {expected}")
