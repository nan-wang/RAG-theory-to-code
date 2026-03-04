"""Naive RAG - 索引构建流程

本脚本演示 RAG 系统中索引构建（Indexing）的完整流程：
1. 加载文档：从指定目录批量读取文本文件
2. 文档分块：使用递归字符分割器将长文档切分为适合检索的小块
3. 向量化存储：通过嵌入模型将文本块转为向量，存入 Chroma 向量数据库

索引构建是 RAG 系统的离线准备阶段，只需执行一次。
构建完成后的向量数据库可供 01_query.py 进行在线检索。
"""

import os
import glob

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量（VECTOR_DB_DIR、COLLECTION_NAME、EMBEDDING_MODEL 等）
dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def load_documents(pathname: str):
    """从指定路径批量加载文本文件。

    使用 glob 模式匹配文件路径，逐个加载为 LangChain Document 对象。

    Args:
        pathname: glob 模式的文件路径，如 "../../data/*.txt"。

    Returns:
        list[Document]: 加载的文档列表，每个文件对应一个 Document。
    """
    doc_list = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        doc_list += loader.load()
    return doc_list


def get_chunks(doc_list: list[Document]):
    """将文档列表分割为较小的文本块。

    使用 RecursiveCharacterTextSplitter 进行分割，该分割器会按照
    段落、句子、字符等层级递归尝试，尽量在语义边界处切分。

    Args:
        doc_list: 待分割的文档列表。

    Returns:
        list[Document]: 分割后的文本块列表，每块保留原文档的元数据。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # 每个文本块的最大字符数
        chunk_overlap=128,  # 相邻文本块之间的重叠字符数，确保上下文连贯
        add_start_index=True,  # 在元数据中记录该块在原文中的起始位置
    )
    return text_splitter.split_documents(doc_list)


docs = load_documents("../../data/*.txt")
print(f"Loaded {len(docs)} documents")

chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")

# 使用 Chroma.from_documents 一步完成向量化和持久化存储：
# 1. 调用嵌入模型将每个文本块转为向量
# 2. 将向量和文档内容存入 Chroma 数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        chunk_size=16,  # 每批发送给嵌入 API 的文档数量
        check_embedding_ctx_length=False,  # 跳过上下文长度检查（兼容第三方 API）
    ),
    persist_directory=VECTOR_DB_DIR,  # 向量数据库的本地持久化目录
    collection_name=COLLECTION_NAME,  # 集合名称，用于区分不同的索引
)
