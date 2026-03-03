# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Write global metric scores to shared scores.json

## Context
`02_calculate_global_metrics.py` computes `precision_score`, `recall_score`, and `f1` but only prints them. The user wants these persisted to `Path(output_path) / "metrics" / "scores.json"`. This file is shared across all three metric scripts (`02_calculate_global_metrics.py`, `02_calculate_retrieval_metrics.py`, `02_calculate_generation_metrics.py`), so the file must be read first (if it exists),...

### Prompt 2

good. make a git add and commit

