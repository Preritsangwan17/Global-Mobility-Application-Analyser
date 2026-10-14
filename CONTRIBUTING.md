# Contributing

Describe the problem or intended behavior in an issue, then make a focused change on a branch.
Include a concrete before/after example and the validation you ran in your pull request.

## Local checks

Run from the repository root with Python 3.11 or 3.12:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Keep generated artifacts, caches, credentials and personal data out of commits. Dataset access and
trained results must be documented separately from software tests. Do not report synthetic test
metrics as real model performance. Keep setup commands and model limitations accurate when code changes.

## Reproduce numerical results

Use `python -m pip install -r requirements-repro.txt` in a fresh virtual environment for the
recorded numerical-library versions. Run software tests separately with `requirements-dev.txt`.
Dataset fingerprints and split seeds are still required to reproduce an evaluation.
