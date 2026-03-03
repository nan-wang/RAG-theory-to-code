# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Reorganize `04_evaluation/` into Three Subfolders

## Context

The `ch05/04_evaluation/` directory currently has all scripts, shared modules, and prompts flat in one directory. We need to organize them into three independent subfolders (`01_extract_keypoints/`, `02_calculate_metrics/`, `03_generate_qa_pairs/`) so each folder can be executed independently (except for data dependencies via CLI args).

## Target Structure

```
ch05/04_evaluation/
├── .en...

### Prompt 2

in @ch05/04_evaluation/03_generate_qa_pairs/03_validate_qa_pairs.py, rename `input_path` to `input_fn`.

### Prompt 3

make a git add and commit

