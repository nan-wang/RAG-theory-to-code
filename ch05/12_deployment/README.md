# 部署

## 简介

本章演示如何将 RAG 系统部署到生产环境。采用前后端分离架构：后端使用 FastAPI 提供 RESTful API，前端使用 Streamlit 构建交互式聊天界面。RAG 流水线基于腾讯云向量数据库实现混合检索（向量 + BM25）和 Jina Reranker 重排序。支持 Docker 容器化一键部署。

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| Docker Compose 配置 | 定义 API 和前端服务的容器编排配置 | [docker-compose.yml](docker-compose.yml) |
| API Dockerfile | 后端 API 服务的容器构建文件 | [api/Dockerfile](api/Dockerfile) |
| API 依赖配置 | 后端 API 的 Python 依赖配置 | [api/pyproject.toml](api/pyproject.toml) |
| API 入口 | FastAPI 应用入口，定义 `/ask` 接口 | [api/src/main.py](api/src/main.py) |
| API 启动脚本 | API 服务的容器启动脚本 | [api/src/entrypoint.sh](api/src/entrypoint.sh) |
| 请求模型 | 定义 API 请求的数据模型 | [api/src/models/query.py](api/src/models/query.py) |
| RAG 索引构建 | 构建腾讯云向量数据库的混合检索索引 | [api/src/rag/rag_index.py](api/src/rag/rag_index.py) |
| RAG 查询流程 | 基于 LangGraph 实现的 RAG 查询流水线 | [api/src/rag/rag_query.py](api/src/rag/rag_query.py) |
| 混合检索实现 | 腾讯云向量数据库的混合检索封装 | [api/src/rag/tcvectordb_hybrid_search.py](api/src/rag/tcvectordb_hybrid_search.py) |
| 生成提示词 | RAG 生成环节的提示词模板 | [api/src/rag/prompts.py](api/src/rag/prompts.py) |
| RAG 工具函数 | 文档加载、分块等工具函数 | [api/src/rag/utils.py](api/src/rag/utils.py) |
| 异步工具 | 异步执行的工具函数 | [api/src/utils/async_utils.py](api/src/utils/async_utils.py) |
| 前端 Dockerfile | 前端服务的容器构建文件 | [frontend/Dockerfile](frontend/Dockerfile) |
| 前端依赖配置 | 前端的 Python 依赖配置 | [frontend/pyproject.toml](frontend/pyproject.toml) |
| 前端主页面 | Streamlit 聊天界面的实现 | [frontend/src/main.py](frontend/src/main.py) |
| 前端启动脚本 | 前端服务的容器启动脚本 | [frontend/src/entrypoint.sh](frontend/src/entrypoint.sh) |
