# 09_naive_rag - 基础 RAG（检索增强生成）

本目录演示最基础的 RAG（Retrieval-Augmented Generation）流程，即"检索 → 生成"两步管道。

## RAG 概念简介

RAG（检索增强生成）是一种将**外部知识检索**与**大语言模型生成**相结合的技术范式：

1. **检索（Retrieve）**：根据用户问题，从预先构建的知识库中检索相关文档片段
2. **生成（Generate）**：将检索到的文档作为上下文，交给大语言模型生成最终回答

相比纯 LLM 问答，RAG 能够利用最新的、领域特定的知识，减少模型幻觉。

## 前置条件

- Python 3.10
- 已安装 `requirements.txt` 中的依赖
- 已配置 `.env` 文件（参考 `../.env.example`）

## 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `OPENAI_API_KEY` | LLM 和嵌入模型的 API 密钥 | `sk-xxx` |
| `OPENAI_API_BASE` | API 基础地址 | `https://api.siliconflow.cn/v1` |
| `VECTOR_DB_DIR` | Chroma 向量数据库的本地存储目录 | `./data_chroma` |
| `COLLECTION_NAME` | 向量数据库中的集合名称 | `olympic_games` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `text-embedding-3-small` |

## 运行步骤

### 第一步：构建索引

```bash
python 01_index.py
```

该脚本执行以下操作：
1. 从 `../../data/*.txt` 加载奥运会文档（24 个文本文件）
2. 使用 `RecursiveCharacterTextSplitter` 将文档分割为 512 字符的文本块（重叠 128 字符）
3. 调用嵌入模型将文本块转为向量
4. 将向量和文档内容持久化到 Chroma 数据库

预期输出：
```
Loaded 24 documents
Split the documents into XXX chunks
```

### 第二步：检索问答

```bash
python 01_query.py
```

该脚本执行以下操作：
1. 连接已构建的 Chroma 向量数据库
2. 根据用户问题检索最相关的 4 个文档块
3. 将检索到的文档作为上下文，交给 Qwen 模型生成回答

预期输出：
```
{'answer': '2024年巴黎奥运会的开幕式于2024年7月26日晚上7点30分举行。',
 'context': [...],
 'question': '2024年巴黎奥运会的开幕式是哪一天?'}
```

## 核心概念

### 文档分块（Chunking）

`RecursiveCharacterTextSplitter` 按照段落 → 句子 → 字符的优先级递归分割文档，尽量在语义边界处切分：

- `chunk_size=512`：每个文本块最大 512 字符
- `chunk_overlap=128`：相邻块之间重叠 128 字符，保持上下文连贯性

### 向量检索（Vector Retrieval）

使用嵌入模型将文本转为高维向量，通过余弦相似度找到与查询最相关的文档块。`search_kwargs={"k": 4}` 表示返回最相似的 4 个结果。

### StateGraph 管道

使用 LangGraph 的 `StateGraph` 将 `retrieve` 和 `generate` 两个函数编排为顺序管道，状态在节点间自动传递。

## 常见问题

**Q: 运行 01_query.py 报错 "Collection not found"？**
A: 请先运行 `01_index.py` 构建向量索引。

**Q: 嵌入模型调用报错？**
A: 检查 `.env` 中的 `OPENAI_API_KEY` 和 `OPENAI_API_BASE` 是否正确配置。

**Q: 如何更换嵌入模型？**
A: 修改 `.env` 中的 `EMBEDDING_MODEL`，注意更换模型后需要重新运行 `01_index.py` 重建索引。
