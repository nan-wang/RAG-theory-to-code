# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Use separate prompt modules for each metric in `02_calculate_generation_metrics.py`

## Context

Same refactoring pattern already applied to `02_calculate_retrieval_metrics.py` and `02_calculate_global_metrics.py` — replace the single legacy `prompts.keypoints_verify_prompt` import with two separate prompt modules so each metric uses the appropriate verification prompt.

## File to modify

`ch05/04_evaluation/02_calculate_generation_metrics.py`

## Change...

