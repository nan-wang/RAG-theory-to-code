# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Rename `input_path` to `input_fn` in `03_rewrite_qa_pairs.py`

## Context
Continuing the argument-style unification across `ch05/04_evaluation` scripts. `03_rewrite_qa_pairs.py` currently uses `--input_path` (required). Rename it to `--input_fn/-i` with `default=None` and a `RuntimeError` if missing, matching the pattern in the other scripts.

## Changes

### 1. `ch05/04_evaluation/03_rewrite_qa_pairs.py`
- Line 62: rename `main` parameter `input_path` → ...

### Prompt 2

good. make a commit

