"""Prompt Optimization RAG - 统一的索引与检索问答脚本

本脚本将索引构建和检索问答合并为一个 CLI 工具：
- --index: 从文本文件构建 Chroma 向量索引
- --query: 使用混合检索（向量 + BM25 + Rerank）并通过优化的提示词生成回答

使用示例:
    # 仅构建索引
    python prompt_optimization.py --index --index_dir data_chroma \
        --collection_name olympic_games --index_input_dir ../data

    # 仅查询
    python prompt_optimization.py --query --index_dir data_chroma \
        --collection_name olympic_games \
        --index_input_dir ../data \
        --query_input_path ../05_chunking/data_eval/keypoints.json \
        --output_dir data_eval 

    # 同时执行索引和查询
    python prompt_optimization.py --index --query --index_dir data_chroma \
        --collection_name olympic_games --index_input_dir ../data \
        --query_input_path ../05_chunking/data_eval/keypoints.json \
        --output_dir data_eval 
"""

import argparse
import asyncio
import json
from argparse import BooleanOptionalAction
from pathlib import Path

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnablePick,
    RunnableConfig,
)
from langchain_openai import ChatOpenAI
from loguru import logger

from utils import load_documents, split_sections, split_chunks, format_docs

dotenv.load_dotenv()

seg = pkuseg.pkuseg()


class Response(BaseModel):
    """LLM 结构化输出模型，包含从上下文中筛选的相关内容和最终回答。"""

    selected_content: str = Field(
        ...,
        description="selected content from the context that is useful to answer the question.",
    )
    answer: str = Field(..., description="the final answer to the question.")


def tokenize_doc(doc_str: str):
    """使用 pkuseg 对文档字符串逐行分词，返回词元列表（BM25 预处理函数）。"""
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        # 跳过空行，对每行进行中文分词
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != ""]
        result += split_tokens
    return result


def get_all_splits(index_input_dir: str):
    """加载指定目录下的所有 txt 文件，按章节切分后返回所有文本块列表。"""
    docs = load_documents(f"{index_input_dir}/*.txt")
    logger.info(f"Loaded {len(docs)} documents from {index_input_dir}")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


prompt_template = ChatPromptTemplate.from_template(
    """You're a helpful AI assistant. Given a user question related to the Olympic Games and some Wikipedia article snippets, answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
Follow the steps,
Step 1: Read the ``Question``.
Step 2: Select the content useful to answer the ``Question`` from ``Context``.
Step 3: Use the selected content from Step 2 to generate an answer.
Use three sentences maximum and keep the answer concise.
------
举例:
Question: "中国在奥运会上有哪些重要历史时刻? "
Context: "[doc_1]article_title: 1996年夏季奥林匹克运动会\\nsection_title: 焦点 香港为最后一次以「香港」和「Hong Kong」名义出席奥林匹克运动会，滑浪风帆选手李丽珊赢得香港历史性首面奥运金牌。\\n\\n[doc_1]article_title: 1984年夏季奥林匹克运动会 section_title: 焦点（社会主义国家里中国、罗马尼亚、南斯拉夫、索马里、贝宁、刚果和莫桑比克参加，这些国家与苏联关系较差） 中华人民共和国自1952年部份参加后，首次全程参与夏季奥运会，许海峰获得了中国也是本届奥运会的首枚金牌，实现了中国零的突破。\\n\\n[doc_2]article_title: 2002年冬季奥林匹克运动会 section_title: 焦点 本届奥运的开幕式比照1992年巴塞隆纳奥运，将开幕式从白天改至晚上举行。 中国在短道速滑女子500米决赛中，杨扬击败了保加利亚的艾芙金妮亚·拉达诺娃和队友王春露，夺得了冠军，为中国自1980年冬季奥林匹克运动会参赛以来首枚金牌。\\n\\n[doc_3]article_title: 1992年夏季奥林匹克运动会 section_title: 焦点 白俄罗斯的体操选手维塔里·谢尔博独自夺得6枚金牌，创下在单届奥运会中取得最多金牌的记录。 棒球首次成为正式奥运会项目，古巴夺得金牌，中国台湾夺得银牌。\\n\\n[doc_4]article_title: 2008年夏季奥林匹克运动会 section_title: summary\\n主办国中华人民共和国以51面金牌成为金牌榜首名，是奥运历史上首个登上金牌榜首的亚洲国家，强化了中国作为体育强国的地位。美国以112面奖牌（36金，39银，37铜）为本届奥运会最多奖牌的国家。"
selected_content: "[doc_1]article_title: 1984年夏季奥林匹克运动会 section_title: 焦点 中华人民共和国自1952年部份参加后，首次全程参与夏季奥运会，许海峰获得了中国也是本届奥运会的首枚金牌，实现了中国零的突破。\\n\\n[doc_2]article_title: 2002年冬季奥林匹克运动会 section_title: 焦点 中国在短道速滑女子500米决赛中，杨扬击败了保加利亚的艾芙金妮亚·拉达诺娃和队友王春露，夺得了冠军，为中国自1980年冬季奥林匹克运动会参赛以来首枚金牌。\\n\\n[doc_4]article_title: 2008年夏季奥林匹克运动会 section_title: summary\\n主办国中华人民共和国以51面金牌成为金牌榜首名，是奥运历史上首个登上金牌榜首的亚洲国家，强化了中国作为体育强国的地位。"
answer: "中国在奥运会上有几个重要的历史时刻。1984年，中华人民共和国首次全程参加夏季奥运会，并由许海峰赢得首枚金牌。2002年，中国选手杨扬在冬季奥运会上夺得首枚金牌。2008年，北京奥运会上，中国作为主办国首次登上金牌榜首，确立了其体育强国地位。"
------
以JSON格式返回结果。JSON对象必须包含以下键：
- 'selected_content'：selected content from the ``Context`` that is useful to answer the ``Question``
- 'answer': the final answer to the ``Question``.

下面是你的任务:

Question: {question}
Context: {context}
"""
)


def index(index_input_dir: str, index_dir: str, collection_name: str):
    """从文本文件构建 Chroma 向量索引。"""
    chunks = get_all_splits(index_input_dir)
    logger.info(f"Split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=JinaEmbeddings(model_name="jina-embeddings-v3"),
        persist_directory=index_dir,
        collection_name=collection_name,
    )
    logger.info(f"Index saved to {index_dir} (collection: {collection_name})")


async def query(
    query_input_path: str,
    index_dir: str,
    collection_name: str,
    index_input_dir: str,
    output_dir: str,
    max_concurrency: int,
):
    """从向量索引检索并生成回答，写入 response.json。"""
    vectorstore = Chroma(
        persist_directory=index_dir,
        embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
        create_collection_if_not_exists=False,
        collection_name=collection_name,
    )
    logger.info(f"{vectorstore._chroma_collection.count()} documents loaded")

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    chunks = get_all_splits(index_input_dir)
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_doc)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus").with_structured_output(
        Response
    )

    # 第一步并行执行检索和透传问题；第二步并行保留上下文/问题并调用 LLM 生成结构化回答
    rag_chain = RunnableParallel(
        context=retriever | format_docs, question=RunnablePassthrough()
    ) | RunnableParallel(
        context=RunnablePick("context"),
        question=RunnablePick("question"),
        response=prompt_template | llm,
    )

    with open(query_input_path, "r") as f:
        qa_pairs = json.load(f)

    inputs = [doc["query"] for doc in qa_pairs]
    logger.info(
        f"Processing {len(inputs)} queries with max_concurrency={max_concurrency}..."
    )

    outputs = await rag_chain.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
    logger.info("Complete generation")

    results = []
    for doc, result in zip(qa_pairs, outputs):
        doc["response"] = {
            "content": result["response"].answer,
            "contexts": [result["context"]],
            "selected_content": result["response"].selected_content,
        }
        results.append(doc)

    output_path = f"{output_dir}/response.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(results)} results to {output_path}")


def main():
    """解析命令行参数，根据标志依次执行索引构建和/或检索问答流程。"""
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
        help="Directory containing *.txt files for indexing and BM25.",
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
        "--max_concurrency", default=2, type=int, help="Batch concurrency for querying."
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
        if not args.index_input_dir:
            parser.error(
                "--index_input_dir is required when --query is set (needed for BM25)."
            )
        asyncio.run(
            query(
                query_input_path=args.query_input_path,
                index_dir=args.index_dir,
                collection_name=args.collection_name,
                index_input_dir=args.index_input_dir,
                output_dir=args.output_dir,
                max_concurrency=args.max_concurrency,
            )
        )


if __name__ == "__main__":
    main()
