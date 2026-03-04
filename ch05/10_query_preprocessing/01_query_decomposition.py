"""
查询分解（Query Decomposition）示例。

将复杂的用户问题拆解为多个独立的子问题，分别检索后合并上下文，
再由 LLM 统一生成答案。适用于需要多步推理的奥运会相关问题。

使用方法：
    python 01_query_decomposition.py \\
        --index_dir ./data_chroma \\
        --collection_name olympic_games \\
        --question "过去5届夏季奥运会都是在哪里举办的?"
"""

import argparse
from pprint import pprint

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

dotenv.load_dotenv()


class State(TypedDict):
    """LangGraph 状态定义，包含原始问题、子问题列表、检索上下文和最终答案。"""

    question: str
    decomposed_questions: List[str]
    context: List[Document]
    answer: str


prompt_template = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
It is May 1st, 2025 today.
Question: {question}
Context: {context}
Answer:
""")

decomposition_prompt_template = ChatPromptTemplate.from_template("""
你是一名国际奥林匹克委员会的专家。你擅长分析关于奥运会的问题。你的任务是判断用户提出的问题是否需要拆解为多个子问题，每个子问题可以被独立回答。今天的日期是2025年5月1日。
必须满足以下要求:
- 每个问题可以被独立回答
- 子问题必须是中文
- 不要做任何解释或者输出任何其他无关信息！
- 直接输出子问题

例子1：
用户问题：列举过去5届夏季奥运会吉祥物的名称?
子问题:
2024年夏季奥运会吉祥物名称是什么？
2020年夏季奥运会吉祥物名称是什么？
2016年夏季奥运会吉祥物名称是什么？
2012年夏季奥运会吉祥物名称是什么？
2008年夏季奥运会吉祥物名称是什么？

用户问题: {question}
Output:""")


def decompose_query(state: State):
    """将原始问题拆解为多个独立子问题。

    调用 LLM，根据 decomposition_prompt_template 将复杂问题拆分为
    若干可独立回答的子问题，每行一个子问题。
    """
    prompt = decomposition_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response_message = llm.invoke(prompt)
    question_list = response_message.content.split("\n")
    return {"decomposed_questions": question_list}


def retrieve(state: State):
    """批量检索每个子问题对应的文档，合并为统一上下文列表。"""
    retrieved_docs = retriever.batch(state["decomposed_questions"])
    results = []
    for l in retrieved_docs:
        results.extend(l)
    return {"context": results}


def generate(state: State):
    """将合并后的上下文传入 LLM，针对原始问题生成最终答案。"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


def main():
    parser = argparse.ArgumentParser(
        description="查询分解 RAG：将复杂问题拆解为子问题后分别检索并生成答案。"
    )
    parser.add_argument(
        "--index_dir",
        required=True,
        type=str,
        help="Chroma 向量数据库的本地存储路径（替代环境变量 VECTOR_DB_DIR）。",
    )
    parser.add_argument(
        "--collection_name",
        required=True,
        type=str,
        help="Chroma 集合名称（替代环境变量 COLLECTION_NAME）。",
    )
    parser.add_argument(
        "--question", required=True, type=str, help="用户输入的查询问题。"
    )
    args = parser.parse_args()

    global llm, retriever

    vector_store = Chroma(
        persist_directory=args.index_dir,
        embedding_function=OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-0.6B",
            chunk_size=16,
            check_embedding_ctx_length=False,
        ),
        create_collection_if_not_exists=False,
        collection_name=args.collection_name,
    )

    llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    graph_builder = StateGraph(State)
    graph_builder.add_node(decompose_query)
    graph_builder.add_node(retrieve)
    graph_builder.add_node(generate)
    graph_builder.add_edge(START, "decompose_query")
    graph_builder.add_edge("decompose_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()

    result = graph.invoke({"question": args.question})
    pprint(result["decomposed_questions"])

    # 示例输出：
    # ['2024年夏季奥运会举办地是哪里？',
    #  '2020年夏季奥运会举办地是哪里？',
    #  '2016年夏季奥运会举办地是哪里？',
    #  '2012年夏季奥运会举办地是哪里？',
    #  '2008年夏季奥运会举办地是哪里？']


if __name__ == "__main__":
    main()
