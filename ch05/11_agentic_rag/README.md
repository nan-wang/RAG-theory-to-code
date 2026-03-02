# 基于智能体的RAG

## 简介

本章基于 LangGraph 构建智能体 RAG 工作流。系统通过多步迭代的方式完成信息检索与总结：生成搜索查询、执行检索（支持本地向量检索和网络搜索）、增量总结、反思评估知识缺口，并决定是否继续搜索。通过循环和条件路由实现自主决策的检索增强生成。

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 状态定义 | 定义智能体 RAG 工作流的状态数据结构 | [src/state.py](src/state.py) |
| 配置 | 定义搜索深度、模型选择等可配置参数 | [src/configuration.py](src/configuration.py) |
| 提示词 | 查询生成、总结和反思等环节的提示词模板 | [src/prompts.py](src/prompts.py) |
| 工作流图 | 基于 LangGraph StateGraph 实现的完整智能体 RAG 工作流（含混合检索） | [src/graph.py](src/graph.py) |
| 工具函数 | 搜索结果格式化、去重、thinking token 处理等工具函数 | [src/utils.py](src/utils.py) |

## 运行

```bash
cd src
python graph.py
```