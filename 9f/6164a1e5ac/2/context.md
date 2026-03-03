# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Modify ch05/11_agentic_rag/graph.py

## Context
The user wants to update `graph.py` in the agentic RAG module to:
1. Replace the `JinaEmbeddings` model with `OpenAIEmbeddings` using a hardcoded Qwen model name
2. Accept `index_dir` and `collection_name` via `argparse` (instead of hardcoded values or env vars)
3. Add Chinese docstrings throughout for readability

Currently, `graph.py` initializes the vectorstore and retriever at module level using hardcoded ...

### Prompt 2

in @ch05/11_agentic_rag/graph.py, when `load_documents`, use `--index_input_dir` to get the input from the cli argument. Load all the .txt files from `index_input_dir`.

### Prompt 3

make a commit

