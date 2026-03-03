# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Rename `--input-dir` to `--input-fn` in `evaluate.sh`

## Context
Continuing argument-style unification. `evaluate.sh` currently takes `--input-dir` (a directory) and derives `RAG_RESPONSE_INPUT_PATH="${INPUT_BASE_DIR}/response.json"`. The user wants `--input-fn` to accept the file path directly, eliminating the derivation.

## File: `ch05/04_evaluation/evaluate.sh`

### Changes
1. **Usage text** (lines 14, 17, 23): `--input-dir <path>` → `--input-fn <pat...

### Prompt 2

good. make a commit

