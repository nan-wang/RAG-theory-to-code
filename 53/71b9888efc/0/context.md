# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Add argparse CLI and Chinese Docstrings to ch05/10_query_preprocessing

## Context

Three scripts in `ch05/10_query_preprocessing/` currently read `VECTOR_DB_DIR` and `COLLECTION_NAME` from environment variables at module-level. The goal is to:
1. Replace those env-var reads with `argparse` CLI arguments (`--index_dir`, `--collection_name`) to match the established pattern in ch05/06, 08, 09.
2. Add Chinese docstrings and inline explanations to make the cod...

### Prompt 2

in these three files, replace the `JinaEmbeddings` with `OpenAIEmbeddings` and use the model "Qwen/Qwen3-Embedding-0.6B". Don't read the model from the environment variables.

### Prompt 3

when using `OpenAIEmbeddings`, we must set the other two parameters,        chunk_size=16, check_embedding_ctx_length=False,

### Prompt 4

update @ch05/10_query_preprocessing/README.md. Create a new section with the name ``运行步骤`` to incluce the example shell scripts for running the python scripts. Add minimal explaination in Chinese.

### Prompt 5

make a commit

