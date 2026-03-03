# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Unify argument style for global and retrieval metrics scripts

## Context
Continuing the pattern established in `01_extract_keypoints.py` and `02_calculate_generation_metrics.py`: converting `input_fn` from a positional argument to an optional `--input_fn/-i` with validation, and renaming `output_path` to `output_dir`.

## Changes

### 1. `ch05/04_evaluation/02_calculate_global_metrics.py`
- Remove positional `input_fn` (line 23: `parser.add_argument('input...

### Prompt 2

make a commit

