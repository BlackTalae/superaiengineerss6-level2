# SuperAIEngineerSS6_Level2

Repository for Super AI Engineer Season 6, Level 2.

## Thai Call Center ASR

Current best submission:

`outputs/submission_large_v3_au_quality_ratio_09.csv`

Best public score observed: `16.36492`.

The raw competition audio, local Python environments, downloaded model/runtime
packages, logs, checkpoints, and archived experiments are intentionally ignored
by git. To rebuild the current best submission from the retained intermediate
CSV files:

```bash
python build_au_quality_filters.py
```
