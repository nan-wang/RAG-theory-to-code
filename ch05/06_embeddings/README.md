# 优化检索模块：嵌入模型

## 简介

本章介绍不同嵌入模型的使用与微调方法。内容涵盖 Jina Embeddings、BGE、本地 HuggingFace 模型等多种嵌入方案的接入，以及如何通过生成训练数据和 SentenceTransformer 框架对嵌入模型进行领域微调，以提升检索效果。

## 环境配置检查

```bash
cd ch05/06_embeddings
python test_env_setup.py
```

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 5.6.1 使用 Jina Embeddings | 使用 Jina Embeddings v3 构建向量索引并执行 RAG 流程 | [01_use_jina_embeddings.py](01_use_jina_embeddings.py) |
| 5.6.1 使用 BGE Embeddings | 通过硅基流动 API 使用 BGE-M3 嵌入模型 | [01_use_bge_embeddings.py](01_use_bge_embeddings.py) |
| 5.6.1 使用本地嵌入模型 | 使用 HuggingFace 本地多语言嵌入模型 | [01_use_local_embeddings.py](01_use_local_embeddings.py) |
| 5.6.2 生成训练三元组 | 基于文档自动生成（query, positive, negative）训练三元组 | [02_generate_qa_triplets.py](02_generate_qa_triplets.py) |
| 5.6.2 微调嵌入模型 | 使用 SentenceTransformer 和 MultipleNegativesRankingLoss 微调 Jina Embeddings v3 | [02_finetune_embeddings.py](02_finetune_embeddings.py) |
| 5.6.2 使用微调后的嵌入模型 | 加载微调后的 Jina Embeddings v3 模型进行检索 | [02_use_finetuned_embeddings.py](02_use_finetuned_embeddings.py) |
| 5.6.2 合成数据提示词 | 生成训练数据的提示词模板 | [synthetic_data_prompt.py](synthetic_data_prompt.py) |
| 工具函数 | 文档加载、分块、格式化等通用工具函数 | [utils.py](utils.py) |

## 运行步骤

### 5.6.1 使用嵌入模型

三个嵌入模型脚本（Jina / BGE / 本地）共用相同的 CLI，支持单独或同时执行索引与查询。

```bash
# 构建向量索引
python 01_use_jina_embeddings.py --index \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --index_input_dir ../data

# 执行 RAG 查询
python 01_use_jina_embeddings.py --query \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --query_input_path ../05_chunking/data_eval/keypoints.json \
    --output_dir data_eval/

# 同时执行索引与查询
python 01_use_jina_embeddings.py --index --query \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --index_input_dir ../data \
    --query_input_path ../05_chunking/data_eval/keypoints.json \
    --output_dir data_eval/
```

将 `01_use_jina_embeddings.py` 替换为 `01_use_bge_embeddings.py` 或 `01_use_local_embeddings.py` 即可切换嵌入模型，参数不变。

### 5.6.2 生成训练三元组

从已有 Chroma 索引中随机采样文档，调用 LLM 生成 `(query, positive, negative)` 三元组，输出 `qa_pairs.json` 和 `qa_triplets.json`。

```bash
# 处理所有文档（默认）
python 02_generate_qa_triplets.py \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --output_dir data_finetuning

# 仅使用前 100 条文档
python 02_generate_qa_triplets.py \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --output_dir data_finetuning \
    -n 100
```

### 5.6.2 微调嵌入模型

将多次生成的三元组文件合并为 `qa_triplets.merged.json` 后，运行微调脚本。

```bash
# 单 GPU
python 02_finetune_embeddings.py

# 多 GPU（使用 accelerate）
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 02_finetune_embeddings.py
```

微调后的模型保存至 `models/jina-embeddings-v3-olympics/v1`。

### 5.6.2 使用微调后的嵌入模型

与 5.6.1 用法相同，额外通过 `--model_path` 指定微调模型路径。

```bash
# 构建索引并查询
python 02_use_finetuned_embeddings.py --index --query \
    --index_dir data_chroma \
    --collection_name olympic_games \
    --index_input_dir ../data \
    --query_input_path ../05_chunking/data_eval/keypoints.json \
    --output_dir data_eval/ \
    --model_path models/jina-embeddings-v3-olympics/v1
```
