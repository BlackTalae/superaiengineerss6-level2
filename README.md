# Thai Math VQA Pipeline

This branch contains the final local-only pipeline used for the Thai Math VQA
experiment. It does not call any hosted inference API.

## Current Best Submission

Submit:

```text
submission_rules_overlay_normalized.csv
```

This file is generated from the baseline `submission.csv` plus a small set of
high-confidence symbolic/rule overrides. It scored better than the previous LLM
attempts in public leaderboard testing.

## Files Used

- `ocr_tha_eng.json`: cached Tesseract Thai/English OCR for all images.
- `ocr_easyocr_all.json`: cached EasyOCR Thai/English OCR for all images.
- `ocr_paddleocr_all.json`: cached PaddleOCR Thai OCR for all images.
- `eval_normalizer.py`: local approximation of the competition normalizer.
- `rule_solver.py`: deterministic symbolic/rule solver over OCR text.
- `canonicalize_submission.py`: converts answers to normalizer-friendly form.
- `local_validate.py`: local validation helpers for train-set experiments.
- `batch_ocr.py`: regenerates OCR caches if needed.
- `make_submission.py`: old OCR nearest-neighbor baseline that produced
  `submission.csv`, used as the base file for rule overlay.

## Reproduce The Final Submission

The OCR files are already cached. To rebuild the final submission from the
baseline and rules:

```powershell
python rule_solver.py --submit --base submission.csv --out submission_rules_overlay.csv
python canonicalize_submission.py submission_rules_overlay.csv --out submission_rules_overlay_normalized.csv
```

To inspect rule behavior on train:

```powershell
python rule_solver.py
```

To run local validation helpers:

```powershell
python local_validate.py --inspect
```

## Regenerate OCR Caches

Only needed if OCR JSON files are missing.

```powershell
python batch_ocr.py --engine easyocr --split all --out ocr_easyocr_all.json --save-every 10
```

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$env:FLAGS_enable_pir_api='0'
python batch_ocr.py --engine paddleocr --split all --out ocr_paddleocr_all.json --save-every 10
```

PaddleOCR on this Windows CPU needed MKLDNN disabled in code; `batch_ocr.py`
already sets `enable_mkldnn=False`.

## Notes

VLM attempts were removed from the final pipeline because local validation and
leaderboard tests showed they were not useful in this environment. The final
approach is intentionally conservative: keep the baseline where uncertain, and
override only text-only problems that deterministic rules solve confidently.
