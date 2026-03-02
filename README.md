# 检索增强生成：理论与实践

本代码仓库是《检索增强生成：理论与实践》(ISBN 978-7-121-51952-9)的官方代码仓库。

![8757b33beb051abf59b398fa7bf90226](https://github.com/user-attachments/assets/65fbffe8-66f7-46bf-a15f-53f250b0fa8e)

## 简介

本书系统性地介绍了检索增强生成（Retrieval-Augmented Generation, RAG）技术的核心原理与工程实践。书中从基础概念出发，逐步深入到文档分块、嵌入模型、混合检索、重排序、生成优化、查询预处理、智能体 RAG 等进阶主题，并最终演示如何将 RAG 系统部署到生产环境。代码示例基于 LangChain 和 LangGraph 框架，使用奥运会数据作为示例数据集。

## 环境配置

运行本仓库中的示例代码前，请先完成环境配置，包括 Python 虚拟环境的创建、依赖安装和环境变量的设置。

详细步骤请参考：[环境配置指南](setup/README.md)

## 内容提纲

| 标题 | 描述 | 文件夹 |
| --- | --- | --- |
| LangChain 核心与 LangGraph | 介绍 LangChain 的核心概念（Runnable、提示模板、输出解析器）和 LangGraph 的状态图编程模型，并实现一个基础 RAG 流程 | [03_langchain_core_and_langgraph](ch05/03_langchain_core_and_langgraph) |
| RAG 评估 | 构建 RAG 系统的评估流水线，包括合成问答对生成、检索指标（精确率、召回率）和生成指标（忠实度、幻觉率等）的计算 | [04_evaluation](ch05/04_evaluation) |
| 文档分块 | 探索不同的文档分块策略，包括针对中文文本的按章节分块方法 | [05_chunking](ch05/05_chunking) |
| 嵌入模型 | 介绍嵌入模型的使用与微调，涵盖 Jina Embeddings、BGE 等多语言嵌入模型 | [06_embeddings](ch05/06_embeddings) |
| 混合检索 | 实现向量相似度检索与 BM25 关键词检索的混合检索策略，提升检索质量 | [07_hybrid_indexing](ch05/07_hybrid_indexing) |
| 重排序 | 使用 Jina Reranker 对检索结果进行重排序，优化检索结果的相关性排序 | [08_reranking](ch05/08_reranking) |
| 生成优化 | 介绍提示工程优化技术，提升 RAG 系统的生成质量 | [09_generative_optimization](ch05/09_generative_optimization) |
| 查询预处理 | 实现查询分解、查询路由和回退提示等查询预处理技术 | [10_query_preprocessing](ch05/10_query_preprocessing) |
| 智能体 RAG | 基于 LangGraph 构建智能体 RAG 工作流，支持多步检索、反思和迭代搜索 | [11_agentic_rag](ch05/11_agentic_rag) |
| 部署 | 使用 FastAPI 和 Streamlit 构建 RAG 应用的生产级部署方案，支持 Docker 容器化部署 | [12_deployment](ch05/12_deployment) |
