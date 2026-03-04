"""Chunking ZH Section RAG - 统一的索引与检索问答脚本

本脚本将索引构建和检索问答合并为一个 CLI 工具：
- --index: 从文本文件按中文章节切分后构建 Chroma 向量索引
- --query: 从向量索引检索并生成回答

使用示例:
    # 仅构建索引
    python 01_chunking_zh_section.py --index --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../data

    # 仅查询
    python 01_chunking_zh_section.py --query --index_dir data_chroma --collection_name olympic_games \
        --query_input_path data_eval/keypoints.json \
        --output_dir data_eval/

    # 同时执行索引和查询
    python 01_chunking_zh_section.py --index --query --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../data \
        --query_input_path data_eval/keypoints.json \
        --output_dir data_eval
"""

import argparse
import asyncio
import json
import re
from argparse import BooleanOptionalAction
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import List, TypedDict

from utils import load_documents

dotenv.load_dotenv()


class State(TypedDict):
    """RAG 工作流的共享状态，包含问题、检索上下文和生成答案。"""

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
    """返回基于 Qwen3-Embedding-0.6B 的 OpenAI 兼容嵌入模型实例。"""
    return OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        chunk_size=16,
        check_embedding_ctx_length=False,
    )


def split_sections(doc: Document):
    """按维基百科式 == 标题 == 语法将文档切分为各章节，保留原始元数据。"""
    text = doc.page_content
    sections = []
    pattern = r"(==+)(.*?)==+\s*([^=]*)"

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    matches = re.finditer(pattern, text, re.DOTALL)

    cur_metadata = doc.metadata.copy()
    for match in matches:
        level = (
            len(match.group(1)) - 1
        )  # Determine the section level by the number of '='
        section_title = match.group(2).strip()
        content = match.group(3).strip()

        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
        section_counters[level] += 1

        cur_metadata["title"] = section_title
        sections.append(Document(page_content=content, metadata=cur_metadata))
    return sections


def split_chunks(docs):
    """将章节文档列表按中文句子边界进一步切分为固定大小的文本块。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["。", "！", "？", "\?", "\n\n", "\n", "\n\n\n"],
        is_separator_regex=True,
        keep_separator="end",
    )
    return text_splitter.split_documents(docs)


def index(index_input_dir: str, index_dir: str, collection_name: str):
    """从文本文件按中文章节切分后构建 Chroma 向量索引。"""
    docs = load_documents(f"{index_input_dir}/*.txt")
    logger.info(f"Loaded {len(docs)} documents from {index_input_dir}")

    chunks = []
    for doc in docs:
        sections = split_sections(doc)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    logger.info(f"Split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=index_dir,
        collection_name=collection_name,
    )
    logger.info(f"Index saved to {index_dir} (collection: {collection_name})")


def retrieve(state: State):
    """从向量数据库中检索与问题最相关的文档。"""
    retrieved_docs = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    ).invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    """根据检索到的上下文文档调用 LLM 生成答案。"""
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

    output_path = Path(output_dir) / "response.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(results)} results to {output_path}")


def main():
    """解析命令行参数并按需执行索引或查询流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        dest="do_index",
        action=BooleanOptionalAction,
        default=False,
        help="Run indexing step.",
    )
    parser.add_argument(
        "--query",
        dest="do_query",
        action=BooleanOptionalAction,
        default=False,
        help="Run query step.",
    )
    parser.add_argument(
        "--index_dir", required=True, type=str, help="Chroma persist directory."
    )
    parser.add_argument(
        "--collection_name", required=True, type=str, help="Chroma collection name."
    )
    parser.add_argument(
        "--index_input_dir",
        default=None,
        type=str,
        help="Directory containing *.txt files for indexing.",
    )
    parser.add_argument(
        "--query_input_path",
        default=None,
        type=str,
        help="Path to input JSON file with queries.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Directory for response.json output.",
    )
    parser.add_argument(
        "--max_concurrency", default=8, type=int, help="Batch concurrency for querying."
    )
    args = parser.parse_args()

    if not args.do_index and not args.do_query:
        parser.error("At least one of --index or --query must be specified.")

    if args.do_index:
        if not args.index_input_dir:
            parser.error("--index_input_dir is required when --index is set.")
        index(
            index_input_dir=args.index_input_dir,
            index_dir=args.index_dir,
            collection_name=args.collection_name,
        )

    if args.do_query:
        if not args.query_input_path:
            parser.error("--query_input_path is required when --query is set.")
        if not args.output_dir:
            parser.error("--output_dir is required when --query is set.")
        asyncio.run(
            query(
                query_input_path=args.query_input_path,
                index_dir=args.index_dir,
                collection_name=args.collection_name,
                output_dir=args.output_dir,
                max_concurrency=args.max_concurrency,
            )
        )


if __name__ == "__main__":
    main()
