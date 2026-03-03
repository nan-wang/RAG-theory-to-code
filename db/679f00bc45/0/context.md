# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Use separate prompt modules for precision vs recall in `02_calculate_retrieval_metrics.py`

## Context

Same refactoring as was done for `02_calculate_global_metrics.py` — replace the single legacy `prompts.keypoints_verify_prompt` import with two separate prompt modules so precision and recall use different verification prompts.

## File to modify

`ch05/04_evaluation/02_calculate_retrieval_metrics.py`

## Changes

### 1. Replace import (line 11)

**Remo...

