# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Modify rag_index.py

## Context
`ch05/12_deployment/api/src/rag/rag_index.py` is a production indexing script that loads Olympic Games documents, generates hybrid embeddings, and upserts them into TencentVectorDB. Currently it hardcodes the data glob path inside `get_all_splits()`. The goals are:
1. Accept `--index_input_dir` as a CLI argument and pass it to `load_documents`
2. Add Chinese docstrings and inline comments for readability
3. After indexing, lo...

### Prompt 2

`--index_input_dir` accepts a folder and we index all the .txt files in this folder. Modify rag_index.py accordingly.

### Prompt 3

make a commit

