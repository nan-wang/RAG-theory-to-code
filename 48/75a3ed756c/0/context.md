# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Convert evaluation scripts to async batch processing

## Context

Four evaluation scripts in `ch05/04_evaluation/` use sequential `chain.invoke()` / `lc_chain.invoke()` loops with `tqdm`. The user wants to convert them all to async `abatch` following the pattern in `03_generate_qa_pairs.py`: `async def main(...)` → `await chain.abatch(inputs, config=RunnableConfig(max_concurrency=...))` → `asyncio.run(main(...))`.

The shared utility `verify_keypoints()...

### Prompt 2

[Request interrupted by user for tool use]

### Prompt 3

Don't run the python codes in this project because it requires Python 3.10 which is conflicts with claude.

### Prompt 4

Focus on @ch05/04_evaluation/02_calculate_global_metrics.py. In line 12, we should not import the same ``SYSTEM_PROMPT`` and ``USER_PROMPT`` from one single module ``prompts.keypoint_verify_prompt`` which is a legacy module and is not longer existing. Instead, 
- import ``SYSTEM_PROMPT`` and ``USER_PROMPT`` from ``prompts.groundtruth_keypoints_verify_prompt`` as ``GT_SYSTEM_PROMPT`` and ``GT_USER_PROMPT``
- import ``SYSTEM_PROMPT`` and ``USER_PROMPT`` from ``prompts.answer_keypoints_verify_promp...

### Prompt 5

[Request interrupted by user for tool use]

