# 使用LangChain搭建RAG系统

## 简介

本章介绍 LangChain 框架的核心概念和 LangGraph 的状态图编程模型。内容涵盖 Runnable 接口的同步与异步调用方式、ChatModel 的使用、提示模板与输出解析器，以及 LangGraph 中状态图（StateGraph）的构建方法。最后通过一个完整的 Naive RAG 示例，演示如何将这些组件组合起来构建一个基础的检索增强生成系统。

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| Runnable invoke | 使用 RunnableGenerator 的 invoke 方法进行同步调用 | [02_runnable_invoke.py](02_runnable_invoke.py) |
| Runnable batch | 使用 RunnableGenerator 的 batch 方法进行批量调用 | [02_runnable_batch.py](02_runnable_batch.py) |
| Runnable stream | 使用 RunnableGenerator 的 stream 方法进行流式输出 | [02_runnable_stream.py](02_runnable_stream.py) |
| Runnable invoke（异步） | 使用 ainvoke 方法进行异步调用 | [02_runnable_invoke_async.py](02_runnable_invoke_async.py) |
| Runnable batch（异步） | 使用 abatch 方法进行异步批量调用 | [02_runnable_batch_async.py](02_runnable_batch_async.py) |
| Runnable stream（异步） | 使用 astream 方法进行异步流式输出 | [02_runnable_stream_async.py](02_runnable_stream_async.py) |
| ChatModel 基本使用 | 调用 OpenAI 和硅基流动的大语言模型 | [03_chatmodel.py](03_chatmodel.py) |
| ChatModel 详细输出 | 通过硅基流动调用 DeepSeek-R1 并查看完整的响应结构 | [03_chatmodel_siliconflow.py](03_chatmodel_siliconflow.py) |
| ChatModel 速率限制 | 使用 InMemoryRateLimiter 控制模型调用频率 | [03_chatmodel_with_ratelimiter.py](03_chatmodel_with_ratelimiter.py) |
| 输出解析器 | 使用 JsonOutputParser 将模型输出解析为结构化 JSON | [03_output_parser.py](03_output_parser.py) |
| 提示模板 | 使用 PromptTemplate 构建动态提示 | [05_prompt_template.py](05_prompt_template.py) |
| 消息占位符 | 使用 MessagesPlaceholder 在提示中插入多轮对话消息 | [05_messages_placeholder.py](05_messages_placeholder.py) |
| StateGraph 基础 | 定义状态和节点，构建简单的 LangGraph 状态图 | [06_graph_state_node.py](06_graph_state_node.py) |
| 状态累加器 | 使用 Annotated 和 add 操作符实现状态的累加更新 | [06_state_add.py](06_state_add.py) |
| 消息状态管理 | 使用 add_messages 管理对话消息列表 | [06_state_add_messages.py](06_state_add_messages.py) |
| 条件边 | 使用 add_conditional_edges 根据状态动态路由到不同节点 | [06_add_conditional_edges.py](06_add_conditional_edges.py) |
| 动态节点 | 使用 Send 动态创建并行节点 | [06_send_for_adding_nodes_dynamically.py](06_send_for_adding_nodes_dynamically.py) |
| Document 与向量存储 | 使用 Document、Embedding 和 InMemoryVectorStore 进行相似度搜索 | [06_document_embedding_vectorstore.py](06_document_embedding_vectorstore.py) |
| 检查点 | 使用 MemorySaver 保存和查看图的执行历史 | [08_checkpoint_memory_saver.py](08_checkpoint_memory_saver.py) |
| Naive RAG - 索引构建 | 加载文档、分块并构建 Chroma 向量索引 | [10_naive_rag/01_index.py](10_naive_rag/01_index.py) |
| Naive RAG - 检索问答 | 基于 LangGraph 实现完整的检索-生成 RAG 流程 | [10_naive_rag/01_query.py](10_naive_rag/01_query.py) |
