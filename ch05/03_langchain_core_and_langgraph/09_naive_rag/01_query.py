"""Naive RAG - 检索问答流程

本脚本演示最基础的 RAG（检索增强生成）问答流程：
1. 检索（Retrieve）：根据用户问题从向量数据库中检索相关文档
2. 生成（Generate）：将检索到的文档作为上下文，交给 LLM 生成回答

整个流程使用 LangGraph StateGraph 编排，形成 retrieve -> generate 的顺序管道。
运行本脚本前，请先执行 01_index.py 构建向量索引。
"""

import os
from pprint import pprint

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# 加载环境变量
dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


class State(TypedDict):
    """RAG 管道的状态定义。

    Attributes:
        question: 用户输入的问题。
        context: 从向量数据库检索到的相关文档列表。
        answer: LLM 生成的回答。
    """
    question: str
    context: List[Document]
    answer: str


# 连接已有的 Chroma 向量数据库（由 01_index.py 构建）
vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        chunk_size=16,
        check_embedding_ctx_length=False
    ),
    create_collection_if_not_exists=False,  # 不自动创建，确保索引已存在
    collection_name=COLLECTION_NAME)

# 定义 RAG 提示模板：指导 LLM 基于检索到的上下文回答问题
prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
)

llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")


def retrieve(state: State):
    """检索节点：根据问题从向量数据库中检索相关文档。

    使用 similarity 搜索策略，返回与问题最相似的 k=4 个文档块。

    Args:
        state: 当前管道状态，包含用户问题。

    Returns:
        dict: 包含检索到的文档列表（context）。
    """
    retrieved_docs = (
        vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}).invoke(state["question"]))
    return {"context": retrieved_docs}


def generate(state: State):
    """生成节点：基于检索到的上下文生成回答。

    将所有检索到的文档内容拼接后填入提示模板，交给 LLM 生成最终回答。

    Args:
        state: 当前管道状态，包含问题和检索到的上下文。

    Returns:
        dict: 包含 LLM 生成的回答（answer）。
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


# 使用 StateGraph 构建 RAG 管道：retrieve -> generate
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

query = "2024年巴黎奥运会的开幕式是哪一天?"
result = graph.invoke({"question": query})
pprint(result)

# 输出
# {
#   "answer": "2024年巴黎奥运会的开幕式于2024年7月26日晚上7点30分举行。",
#   "context": [
#     {
#       "id": "b6d9...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 2567
#       },
#       "page_content": "开幕式于欧洲中部时间2024年7月26日晚..."
#     },
#     {
#       "id": "768b...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 0
#       },
#       "page_content": "2024年夏季奥林匹克运动会 （英语..."
#     },
#     {
#       "id": "dfa1...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 5722
#       },
#       "page_content": "== 赞助商 ==..."
#     },
#     {
#       "id": "94fa...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 2122
#       },
#       "page_content": "=== 火炬 ===..."
#     }
#   ],
#   "question": "2024年巴黎奥运会的开幕式是哪一天?"
# }
