# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Add Async Batch Query and CLI to ch05/06_embeddings

## Context
The four embedding scripts in `ch05/06_embeddings/` use a synchronous `rag_chain.invoke()` loop and have hardcoded paths for inputs/outputs. The goal is to refactor them to follow the pattern established in `ch05/04_evaluation/01_extract_keypoints/naive_rag.py` and `ch05/07_hybrid_indexing/hybrid_rag.py`, adding:
- CLI argument parsing (`--index`, `--query`, and path flags)
- `index()` function...

### Prompt 2

good. make a git commit

