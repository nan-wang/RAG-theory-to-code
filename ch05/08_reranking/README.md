# 优化检索模块：添加重排序

## 简介

本章演示如何在混合检索的基础上添加 Jina Reranker 重排序步骤。使用 LangGraph StateGraph 构建完整的 RAG 流程，通过 ContextualCompressionRetriever 将 EnsembleRetriever（向量检索 + BM25）与 Jina Reranker 组合，优化检索结果的相关性排序。

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 混合检索 + 重排序 RAG | 基于 LangGraph 实现混合检索 + Jina Reranker 重排序的 RAG 流程 | [hybrid_rag_with_reranking.py](hybrid_rag_with_reranking.py) |
