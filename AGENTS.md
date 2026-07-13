# AGENTS.md

## Project Overview

- This repository is `AtriaKit`, a Python 3.10+ framework for atrial ECG analysis and reproducible P-wave feature extraction.
- Keep reusable library code in `ecg/`.
- Keep manuscript-, dataset-, and exploration-specific work in `paper/` unless a task explicitly promotes it into reusable package code.
- Treat `tests/` as the source of truth for reusable behavior.

## Repository Map

- `ecg/`: reusable package code
- `ecg/io/`: ECG/DICOM loading and waveform access
- `ecg/preprocessing/`: signal and annotation preprocessing
- `ecg/features/`: feature implementations
- `ecg/processing/`: datasets and batch pipelines
- `ecg/configs/`: configuration models and defaults
- `ecg/paper/`: scriptable helpers used by paper workflows
- `paper/`: analysis scripts, notebooks, figures, and one-off research utilities
- `tests/`: unit tests for reusable components

## Working Agreements

- Make the smallest change that completely solves the task.
- Preserve existing public APIs, config shapes, and output column names unless the task explicitly calls for a breaking change.
- Prefer fixing shared logic in `ecg/` over copying logic into notebooks or one-off scripts.
- Do not do unrelated refactors, mass reformatting, or broad file moves unless they are necessary for the task.
- Prefer reusable `.py` modules over notebook-only logic when code is likely to be reused.
- Ask before adding new runtime dependencies or changing package metadata.

## Python Conventions

- Follow the style already present in the file you touch; do not impose a new formatter style.
- Keep functions focused and names descriptive.
- Add or preserve type hints when practical, especially in `ecg/`.
- Keep docstrings on reusable public functions and classes.
- For paths, follow the local file's existing style instead of mixing conventions just for preference.

## Research Data And Assets

- Treat DICOMs, CSV annotations, notebooks, and generated figures as research assets; do not rename, delete, or regenerate them unless the task requires it.
- Avoid noisy notebook output churn unless the task is explicitly about the notebook or figure output.
- In reusable package code, prefer repo-relative or config-driven paths over hard-coded machine-specific paths.
- Be careful with changes that could silently alter feature values, annotation alignment, or preprocessing defaults.

## Verification

- Install with `pip install -e .[dev]` when possible. Fallback: `pip install -e .`
- For focused code changes, run the narrowest relevant test file first, for example `pytest tests/test_utils.py -q`.
- Run `pytest -q` when a change affects shared utilities, configs, pipelines, or multiple feature modules.
- If you change behavior in `ecg/`, add or update tests in `tests/` unless there is a strong reason not to.
- If you change `paper/`-only scripts or notebooks, use the lightest realistic validation and clearly report what was not verified.

## Done Means

- The requested behavior is implemented.
- Relevant tests or validation steps were run, or blockers are stated explicitly.
- Public usage docs or examples are updated if package-facing behavior changed.
- The final handoff should mention assumptions, risks, and any skipped verification.

## Collaboration And Safety

- The git worktree may be dirty. Never overwrite or revert unrelated user changes.
- Keep library stability in mind when a task touches both `ecg/` and `paper/`; isolate paper-specific behavior when possible.
- For large or ambiguous tasks, summarize the plan before major edits.
- If you notice repeated friction or the same correction twice, update this file rather than relying on chat memory alone.
