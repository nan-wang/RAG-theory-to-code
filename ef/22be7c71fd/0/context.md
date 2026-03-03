# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Convert `input_fn` from positional to `--input_fn` option in `01_extract_keypoints.py`

## Context

The user wants `input_fn` in `01_extract_keypoints.py` to be an optional `--input_fn`/`-i` argument (same style as `--output_path`/`-o`) instead of a positional argument, with `default=None` and a `RuntimeError` if not provided.

## Changes

### 1. `ch05/04_evaluation/01_extract_keypoints.py` (already done)

Changed:
```python
parser.add_argument('input_fn', ...

