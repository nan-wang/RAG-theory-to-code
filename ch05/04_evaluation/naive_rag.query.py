"""Naive RAG - 检索问答流程

本脚本演示最基础的 RAG（检索增强生成）问答流程：
1. 检索（Retrieve）：根据用户问题从向量数据库中检索相关文档
2. 生成（Generate）：将检索到的文档作为上下文，交给 LLM 生成回答

整个流程使用 LangGraph StateGraph 编排，形成 retrieve -> generate 的顺序管道。
运行本脚本前，请先执行 01_index.py 构建向量索引。
"""

import asyncio
import json
from pathlib import Path

import click
import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import List, TypedDict

# 加载环境变量
dotenv.load_dotenv()


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

llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus")


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


async def main(input_fn: str, index_dir: str, collection_name: str, output_dir: str, max_concurrency: int):
    global vector_store

    # 连接已有的 Chroma 向量数据库（由 01_index.py 构建）
    vector_store = Chroma(
        persist_directory=index_dir,
        embedding_function=OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-0.6B",
            chunk_size=16,
            check_embedding_ctx_length=False
        ),
        create_collection_if_not_exists=False,
        collection_name=collection_name,
    )

    # 使用 StateGraph 构建 RAG 管道：retrieve -> generate
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    with open(input_fn, "r") as f:
        qa_pairs = json.load(f)

    inputs = [{"question": doc["query"]} for doc in qa_pairs]
    logger.info(f"Processing {len(inputs)} queries with max_concurrency={max_concurrency}...")

    outputs = await graph.abatch(inputs, config=RunnableConfig(max_concurrency=max_concurrency))
    logger.info("Complete generation")

    results = []
    for doc, result in zip(qa_pairs, outputs):
        doc["response"] = {
            "content": result["answer"],
            "contexts": [d.page_content for d in result["context"]],
        }
        results.append(doc)

    output_path = f"{output_dir}/response.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(results)} results to {output_path}")


@click.command()
@click.argument("input_fn")
@click.option("--index_dir", required=True, type=str, help="Directory where the vector index is stored.")
@click.option("--collection_name", required=True, type=str, help="Name of the Chroma collection.")
@click.option("--output_dir", required=True, type=str, help="Directory where response.json will be written.")
@click.option("--max_concurrency", default=8, type=int, help="Maximum number of concurrent batch runs.")
def cli(input_fn, index_dir, collection_name, output_dir, max_concurrency):
    asyncio.run(
        main(
            input_fn=input_fn,
            index_dir=index_dir,
            collection_name=collection_name,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        )
    )


if __name__ == "__main__":
    cli()