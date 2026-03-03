"""
退一步提问（Step-Back Prompting）示例。

将具体的用户问题转化为更通用的"退一步"问题，同时检索原始问题和
退一步问题的相关文档，合并上下文后由 LLM 生成答案。适用于需要
背景知识支撑的细节性奥运会问题。

使用方法：
    python 03_query_stepback.py \\
        --index_dir ./data_chroma \\
        --collection_name olympic_games \\
        --question "北京申办过几次奥运会?"
"""
import argparse
from pprint import pprint

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

dotenv.load_dotenv()


class State(TypedDict):
    """LangGraph 状态定义，包含原始问题、退一步问题、检索上下文和最终答案。"""
    question: str
    stepback_question: str
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

# Few-shot 示例：展示如何将具体问题转化为更通用的退一步问题
examples = [
    {
        "input": "巴黎奥运会的吉祥物有什么含义?",
        "output": "介绍巴黎奥运会的吉祥物",
    },
    {
        "input": "2004年奥运会和雅典共同竞争举办权的有哪些国家?",
        "output": "介绍2024年夏季奥运会的申办过程",
    },
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
template = """You are an expert at Olympic Games.\n
Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.\n
The generic step-back question should be answerable given a collection of wikipedia pages that contains \n
the information about the olympic games from 1980 to 2024.\n
Here are a few examples:
"""
stepback_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)


def stepback(state: State):
    """将原始问题转化为更通用的退一步问题。

    调用 LLM，根据 stepback_prompt_template 将细节性问题抽象为
    更宏观的背景性问题，以便检索到更丰富的上下文文档。
    """
    prompt = stepback_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response_message = llm.invoke(prompt)
    return {"stepback_question": response_message.content}


def retrieve(state: State):
    """同时检索原始问题和退一步问题的相关文档，合并为统一上下文列表。"""
    retrieved_docs = retriever.batch([state["question"], state["stepback_question"]])
    results = []
    for l in retrieved_docs:
        results.extend(l)
    return {"context": results}


def generate(state: State):
    """将合并后的上下文传入 LLM，针对原始问题生成最终答案。"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke(
        {
            "question": state["question"], "context": docs_content
        }
    )
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


def main():
    parser = argparse.ArgumentParser(
        description="退一步提问 RAG：将具体问题抽象化后检索更广泛的上下文并生成答案。"
    )
    parser.add_argument(
        "--index_dir", required=True, type=str,
        help="Chroma 向量数据库的本地存储路径（替代环境变量 VECTOR_DB_DIR）。"
    )
    parser.add_argument(
        "--collection_name", required=True, type=str,
        help="Chroma 集合名称（替代环境变量 COLLECTION_NAME）。"
    )
    parser.add_argument(
        "--question", required=True, type=str,
        help="用户输入的查询问题。"
    )
    args = parser.parse_args()

    global llm, retriever

    vector_store = Chroma(
        persist_directory=args.index_dir,
        embedding_function=OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-0.6B",
                                             chunk_size=16,
                                             check_embedding_ctx_length=False),
        create_collection_if_not_exists=False,
        collection_name=args.collection_name,
    )

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus", temperature=0)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})

    graph_builder = StateGraph(State)
    graph_builder.add_node(stepback)
    graph_builder.add_node(retrieve)
    graph_builder.add_node(generate)
    graph_builder.add_edge(START, "stepback")
    graph_builder.add_edge("stepback", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile()

    result = graph.invoke({"question": args.question})
    pprint(result["stepback_question"])

    # 示例输出：
    # '介绍北京申办奥运会的历史'


if __name__ == "__main__":
    main()
