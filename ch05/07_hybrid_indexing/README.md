# 优化检索模块：混合检索

## 简介

本章实现向量相似度检索与 BM25 关键词检索的混合检索策略。使用 pkuseg 进行中文分词以构建 BM25 检索器，结合 Jina Embeddings 向量检索器，通过 EnsembleRetriever 融合两种检索结果，并使用 Jina Reranker 进行重排序，从而提升检索质量。

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 混合检索 RAG | 实现向量检索 + BM25 混合检索 + Jina Rerank 重排序的完整 RAG 流程 | [hybrid_rag.py](hybrid_rag.py) |
| 工具函数 | 文档加载、分块、格式化等通用工具函数 | [utils.py](utils.py) |
