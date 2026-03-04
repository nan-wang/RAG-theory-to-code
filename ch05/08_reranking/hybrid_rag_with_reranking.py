"""Hybrid RAG with Reranking - 统一的索引与检索问答脚本

本脚本将索引构建和检索问答合并为一个 CLI 工具：
- --index: 从文本文件构建 Chroma 向量索引
- --query: 使用混合检索（向量 + BM25）+ Jina 重排序，批量生成回答

使用示例:
    # 仅构建索引
    python hybrid_rag_with_reranking.py --index --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../../data

    # 仅查询
    python hybrid_rag_with_reranking.py --query --index_dir data_chroma --collection_name olympic_games \
        --query_input_path ../05_chunking/data_eval/keypoints.json \
        --output_dir data_eval/

    # 同时执行索引和查询
    python hybrid_rag_with_reranking.py --index --query --index_dir data_chroma --collection_name olympic_games \
        --index_input_dir ../data \
        --output_dir data_eval/
"""

import argparse
import asyncio
import glob
import json
from argparse import BooleanOptionalAction
from pathlib import Path

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import List, TypedDict

dotenv.load_dotenv()

seg = None  # initialized in query()


class State(TypedDict):
    """RAG 图的状态，包含问题、检索到的上下文文档和生成的回答。"""

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
    """返回 Jina Embeddings v3 实例，用于向量化文档和查询。"""
    return JinaEmbeddings(model_name="jina-embeddings-v3")


def load_documents(index_input_dir: str):
    """加载指定目录下所有 txt 文件，返回文档列表。"""
    docs = []
    for file in glob.glob(f"{index_input_dir}/*.txt"):
        loader = TextLoader(file)
        docs += loader.load()
    return docs


def get_chunks(docs: list[Document]):
    """将文档列表按固定大小切分为文本块，返回切分后的文档列表。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


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


def index(index_input_dir: str, index_dir: str, collection_name: str):
    """从文本文件构建 Chroma 向量索引。"""
    docs = load_documents(index_input_dir)
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


async def query(
    query_input_path: str,
    index_dir: str,
    collection_name: str,
    output_dir: str,
    max_concurrency: int,
):
    """使用混合检索 + 重排序从向量索引检索并批量生成回答，写入 response.json。"""
    global seg

    vector_store = Chroma(
        persist_directory=index_dir,
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
        collection_name=collection_name,
    )

    vector_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    ids = vector_store.get()["ids"]
    logger.info(f"Retrieved {len(ids)} documents from the vector store at {index_dir}")
    # 从向量数据库中取回所有文档，重新构建 Document 对象以供 BM25 使用
    raw = {
        k: v
        for k, v in vector_store.get(ids=ids).items()
        if k in ("ids", "metadatas", "documents")
    }
    documents = []
    for t in zip(*raw.values()):
        d = dict(zip(raw, t))
        documents.append(
            Document(page_content=d["documents"], id=d["ids"], metadata=d["metadatas"])
        )

    seg = pkuseg.pkuseg()  # 延迟初始化分词器，避免在主进程中过早占用资源
    bm25_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=tokenize_doc
    )
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    def retrieve(state: State):
        """从混合检索器中获取与问题相关的文档列表。"""
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        """根据检索到的上下文文档，调用 LLM 生成回答。"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = prompt_template.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response_message = llm.invoke(prompt)
        return {"answer": response_message.content}

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
