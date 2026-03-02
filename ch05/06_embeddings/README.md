# 优化检索模块：嵌入模型

## 简介

本章介绍不同嵌入模型的使用与微调方法。内容涵盖 Jina Embeddings、BGE、本地 HuggingFace 模型等多种嵌入方案的接入，以及如何通过生成训练数据和 SentenceTransformer 框架对嵌入模型进行领域微调，以提升检索效果。

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
