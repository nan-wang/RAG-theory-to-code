"""Naive RAG - 统一的索引与检索问答脚本

本脚本将索引构建和检索问答合并为一个 CLI 工具：
- --index: 从文本文件构建 Chroma 向量索引
- --query: 从向量索引检索并生成回答

使用示例:
    # 仅构建索引
    python naive_rag.py --index --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../../data

    # 仅查询
    python naive_rag.py --query --index_dir data_chroma --collection_name olympic_games \
        --query_input_path data_eval/v20250501/keypoints.json \
        --output_dir data_eval/v20250501

    # 同时执行索引和查询
    python naive_rag.py --index --query --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../../data \
        --query_input_path data_eval/v20250501/keypoints.json \
        --output_dir data_eval/v20250501
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

from utils import get_chunks, load_documents

dotenv.load_dotenv()


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


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


def get_embeddings():
    return OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        chunk_size=16,
        check_embedding_ctx_length=False,
    )


def index(index_input_dir: str, index_dir: str, collection_name: str):
    """从文本文件构建 Chroma 向量索引。"""
    docs = load_documents(f"{index_input_dir}/*.txt")
    logger.info(f"Loaded {len(docs)} documents from {index_input_dir}")

    chunks = get_chunks(docs)
    logger.info(f"Split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=index_dir,
        collection_name=collection_name,
    )
    logger.info(f"Index saved to {index_dir} (collection: {collection_name})")


def retrieve(state: State):
    retrieved_docs = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    ).invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


async def query(
    query_input_path: str,
    index_dir: str,
    collection_name: str,
    output_dir: str,
    max_concurrency: int,
):
    """从向量索引检索并生成回答，写入 response.json。"""
    global vector_store

    vector_store = Chroma(
        persist_directory=index_dir,
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
        collection_name=collection_name,
    )

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    with open(query_input_path, "r") as f:
        qa_pairs = json.load(f)

    inputs = [{"question": doc["query"]} for doc in qa_pairs]
    logger.info(
        f"Processing {len(inputs)} queries with max_concurrency={max_concurrency}..."
    )

    outputs = await graph.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
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
@click.option("--index/--no-index", "do_index", default=False, help="Run indexing step.")
@click.option("--query/--no-query", "do_query", default=False, help="Run query step.")
@click.option("--index_dir", required=True, type=str, help="Chroma persist directory.")
@click.option("--collection_name", required=True, type=str, help="Chroma collection name.")
@click.option("--index_input_dir", default=None, type=str, help="Directory containing *.txt files for indexing.")
@click.option("--query_input_path", default=None, type=str, help="Path to input JSON file with queries.")
@click.option("--output_dir", default=None, type=str, help="Directory for response.json output.")
@click.option("--max_concurrency", default=8, type=int, help="Batch concurrency for querying.")
def cli(do_index, do_query, index_dir, collection_name, index_input_dir, query_input_path, output_dir, max_concurrency):
    if not do_index and not do_query:
        raise click.UsageError("At least one of --index or --query must be specified.")

    if do_index:
        if not index_input_dir:
            raise click.UsageError("--index_input_dir is required when --index is set.")
        index(index_input_dir=index_input_dir, index_dir=index_dir, collection_name=collection_name)

    if do_query:
        if not query_input_path:
            raise click.UsageError("--query_input_path is required when --query is set.")
        if not output_dir:
            raise click.UsageError("--output_dir is required when --query is set.")
        asyncio.run(
            query(
                query_input_path=query_input_path,
                index_dir=index_dir,
                collection_name=collection_name,
                output_dir=output_dir,
                max_concurrency=max_concurrency,
            )
        )


if __name__ == "__main__":
    cli()
