"""Document、Embedding 与 VectorStore - RAG 三大核心组件

本脚本演示 LangChain 中 RAG 系统的三大基础组件：
1. Document（文档）：存储文本内容及其元数据的基本单元
2. Embedding（嵌入）：将文本转换为向量表示，用于语义相似度计算
3. VectorStore（向量存储）：存储文档向量并支持相似度检索

这三个组件构成了 RAG 系统中"检索"环节的基础。
"""

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# FakeEmbeddings 是测试用的假嵌入模型，生成随机向量（不具备语义能力）
# 实际使用时应替换为真实的嵌入模型（如 OpenAIEmbeddings、JinaEmbeddings 等）
vector_store = InMemoryVectorStore(embedding=FakeEmbeddings(size=64))

# Document 对象包含两个核心属性：
# - page_content: 文本内容
# - metadata: 元数据字典（如来源、页码等），用于过滤和溯源
document_1 = Document(
    page_content="今天天气很好",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="今天我去了公园",
    metadata={"source": "news"},
)

documents = [document_1, document_2]

# 将文档添加到向量存储：内部会自动调用嵌入模型将文本转为向量
vector_store.add_documents(documents=documents)

# similarity_search：根据查询文本进行语义相似度检索，返回最相似的文档
query = "今天天气怎么样?"
results = vector_store.similarity_search(query)
print(results)
