# SuperAIEngineerSS6_Level2

Repository for Super AI Engineer Season 6, Level 2.

## Thai ASR Pipeline

Current best local submission:

`outputs/submission_large_v3_au_quality_ratio_09.csv`

Score observed locally on the competition leaderboard: `16.36492`.

The large audio dataset, downloaded model packages, experiment archive, logs, and
intermediate outputs are intentionally ignored by git. Keep the Kaggle data at:

`individual-test-thai-call-center-asr/`

The current pipeline scripts are:

- `run_grouped_large_v3.py`
- `run_large_v3_all_variants.py`
- `build_large_v3_ensemble_candidates.py`
- `run_grouped_large_v3_contextual.py`
- `build_contextual_medoid.py`
- `build_singleton_contextual_candidates.py`
- `build_au_quality_filters.py`
