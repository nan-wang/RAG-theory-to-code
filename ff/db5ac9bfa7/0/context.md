# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Rename entry-point functions to `main()` in `naive_rag.py`

## Context
Unifying the main function naming across all Python scripts in `ch05/04_evaluation/`. All scripts already use `main()` except `naive_rag.py`, which uses `cli()`.

### Current state
| Script | Entry function | Status |
|---|---|---|
| `01_extract_keypoints.py` | `main()` | OK |
| `02_calculate_generation_metrics.py` | `main()` | OK |
| `02_calculate_global_metrics.py` | `main()` | OK |
| ...

### Prompt 2

good. make a commit

