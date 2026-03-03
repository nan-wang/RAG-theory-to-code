# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Use separate prompt modules for precision vs recall in `02_calculate_global_metrics.py`

## Context

`02_calculate_global_metrics.py` currently imports `SYSTEM_PROMPT` and `USER_PROMPT` from the legacy module `prompts.keypoints_verify_prompt`, using one shared chain for both precision and recall. The two metrics have different verification directions and should use different prompts:

- **Precision** (`rsp_kp`): response keypoints verified against the groun...

