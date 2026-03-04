"""Agentic RAG 主模块：构建并运行基于混合检索和多轮反思的 LangGraph 工作流。"""

import argparse
import json
from pathlib import Path
from typing import Literal

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

from configuration import Configuration
from prompts import (
    query_writer_instructions,
    get_current_date,
    summarizer_instructions,
    reflection_instructions,
)
from state import SummaryState, SummaryStateInput, SummaryStateOutput
from utils import (
    strip_thinking_tokens,
    get_config_value,
    tavily_search,
    format_sources,
    deduplicate_and_format_sources,
    extract_json_from_markdown,
    load_documents,
    split_chunks,
    split_sections,
)

dotenv.load_dotenv()


def get_all_splits(input_dir: str):
    """加载并切分所有奥运文档。

    从 input_dir 目录下加载所有 .txt 文档，按章节切分后返回文档块列表。

    :param input_dir: 包含 .txt 文档的目录路径
    """
    docs = load_documents(f"{input_dir}/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


seg = pkuseg.pkuseg()


def tokenize_doc(doc_str: str):
    """使用pkuseg对中文文档进行分词。

    逐行处理文本，去除空行后进行中文分词，返回词语列表。
    """
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != ""]
        result += split_tokens
    return result


# 模块级全局变量，在 setup_retriever() 中赋值
retriever = None


def setup_retriever(index_dir: str, collection_name: str, input_dir: str):
    """初始化向量检索器和混合检索器。

    加载Chroma向量数据库，构建BM25检索器，使用集成检索器（向量+BM25）
    并通过JinaRerank进行上下文压缩，最终设置模块级 retriever 全局变量。

    :param index_dir: Chroma向量数据库目录路径
    :param collection_name: 集合名称
    :param input_dir: 包含 .txt 原始文档的目录路径（用于构建BM25索引）
    """
    global retriever

    if Path(index_dir).exists():
        vectorstore = Chroma(
            persist_directory=index_dir,
            embedding_function=OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-0.6B"),
            create_collection_if_not_exists=False,
            collection_name=collection_name,
        )
        print(f"{vectorstore._chroma_collection.count()} documents loaded")
    else:
        raise FileNotFoundError(f"向量数据库目录不存在: {index_dir}")

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    chunks = get_all_splits(input_dir)
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_doc)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )


def generate_query(state: SummaryState, config: RunnableConfig):
    """根据用户主题生成搜索查询。

    使用LLM根据当前搜索主题生成结构化的搜索查询词。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 包含 search_query 的字典
    """
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date, user_query=state.user_query
    )

    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)

    result = llm.invoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"Generate a query for web search:"),
        ]
    )

    content = result.content

    try:
        if content.startswith("```json"):
            content = extract_json_from_markdown(content)
        query = json.loads(content)
        search_query = query["query"]
    except (json.JSONDecodeError, KeyError):
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    return {"search_query": search_query}


def search(state: SummaryState, config: RunnableConfig):
    """执行混合检索（向量+BM25+Jina重排序）。

    使用模块级 retriever 对当前搜索查询进行本地文档检索，
    返回格式化的检索结果。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 包含 sources_gathered、search_loop_count、web_search_results 的字典
    """
    configurable = Configuration.from_runnable_config(config)
    results = retriever.invoke(f"{state.search_query}")
    search_results = {"results": []}
    for idx, doc in enumerate(results):
        content = doc.page_content.replace("\n", " ")
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        search_results["results"].append(
            {"url": url, "content": content, "title": title}
        )
    search_str = deduplicate_and_format_sources(search_results, fetch_full_page=False)
    return {
        "sources_gathered": [format_sources(search_results)],
        "search_loop_count": state.search_loop_count + 1,
        "web_search_results": [
            search_str,
        ],
    }


def web_search(state: SummaryState, config: RunnableConfig):
    """执行网络搜索（Tavily/DuckDuckGo）。

    根据配置选择搜索引擎，对当前查询执行网络搜索并返回格式化结果。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 包含 sources_gathered、search_loop_count、web_search_results 的字典
    """
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_str = ""
    search_results = {}

    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query,
            fetch_full_page=configurable.fetch_full_page,
            max_results=configurable.max_web_search_results,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "duckduckgo":
        ...
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")
    return {
        "sources_gathered": [format_sources(search_results)],
        "search_loop_count": state.search_loop_count + 1,
        "web_search_results": [
            search_str,
        ],
    }


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """根据检索结果生成或更新摘要。

    若已有摘要则结合新检索结果进行更新，否则从头生成摘要。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 包含 running_summary 的字典
    """
    existing_summary = state.running_summary

    most_recent_web_search = state.web_search_results[-1]

    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary> \n\n"
            f"<New Context> \n {most_recent_web_search} \n <New Context> \n\n"
            f"Update the Existing Summary with the New Context on this topic: \n "
            f"<User Input> \n {state.user_query} \n <User Input> \n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_search} \n <Context> \n\n"
            f"Create a Summary using the Context on this topic: \n "
            f"<User Input> \n {state.user_query} \n <User Input> \n\n"
        )

    configurable = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)

    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )
    running_summary = result.content

    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)
    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """反思现有摘要并识别知识盲点。

    分析当前摘要，找出尚未覆盖的知识点，生成后续搜索查询。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 包含 search_query 的字典
    """
    configurable = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)
    result = llm.invoke(
        [
            SystemMessage(
                content=reflection_instructions.format(user_query=state.user_query)
            ),
            HumanMessage(
                content=(
                    f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n "
                    f"And now identify a knowledge gap and generate a follow-up web search query:"
                )
            ),
        ]
    )
    content = result.content
    try:
        if content.startswith("```json"):
            content = extract_json_from_markdown(content)
        reflection_content = json.loads(content)
        query = reflection_content.get("follow_up_query")
        if not query:
            return {"search_query": f"Tell me more about {state.user_query}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {"search_query": f"Tell me more about {state.user_query}"}


def finalize_summary(state: SummaryState):
    """去重来源并生成最终摘要。

    对所有收集到的来源进行去重，将摘要与来源合并为最终输出。

    :param state: 当前图状态
    :return: 包含 running_summary 的字典
    """
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        for line in source.split("\n"):
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources = "\n".join(unique_sources)
    state.running_summary = (
        f"## Summary\n{state.running_summary}\n\n## Sources\n{all_sources}"
    )
    return {"running_summary": state.running_summary}


def route_search(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "search"]:
    """路由决策：继续搜索或结束。

    根据已执行的搜索次数决定是继续循环搜索还是进入最终摘要节点。

    :param state: 当前图状态
    :param config: 运行时配置
    :return: 下一个节点名称
    """
    configurable = Configuration.from_runnable_config(config)
    if state.search_loop_count < configurable.max_web_search_loops:
        return "search"
    else:
        return "finalize_summary"


def build_graph():
    """构建并编译LangGraph工作流图。

    定义节点、边和条件路由，返回编译后的图对象。

    :return: 编译后的 LangGraph 图
    """
    builder = StateGraph(
        SummaryState,
        input=SummaryStateInput,
        output=SummaryStateOutput,
        config_schema=Configuration,
    )
    builder.add_node("generate_query", generate_query)
    builder.add_node("search", search)
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_summary)
    builder.add_node("finalize_summary", finalize_summary)

    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "search")
    builder.add_edge("search", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")
    builder.add_conditional_edges("reflect_on_summary", route_search)
    builder.add_edge("finalize_summary", END)

    return builder.compile(debug=True)


def main():
    """命令行入口：解析参数，初始化检索器，运行图。"""
    parser = argparse.ArgumentParser(description="Agentic RAG 图执行入口")
    parser.add_argument(
        "--index_dir",
        default="data_chroma",
        help="Chroma向量数据库目录路径（默认：data_chroma）",
    )
    parser.add_argument(
        "--collection_name",
        default="olympic_games",
        help="Chroma集合名称（默认：olympic_games）",
    )
    parser.add_argument(
        "--index_input_dir",
        default="../data",
        help="包含 .txt 原始文档的目录路径（默认：../data）",
    )
    args = parser.parse_args()

    setup_retriever(args.index_dir, args.collection_name, args.index_input_dir)
    graph = build_graph()
    output = graph.invoke({"user_query": "2014年冬奥会的吉祥物是什么?是如何被选出的？"})
    print(output["running_summary"])


if __name__ == "__main__":
    main()
