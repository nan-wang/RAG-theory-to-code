"""对通过验证的合成问答对进行改写，以提升问题的清晰度和可回答性。"""

import argparse
import asyncio
import json
from pathlib import Path

import dotenv
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from prompts.rewrite_question_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv()

QUESTION_REWRITE_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
QUESTION_REWRITE_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[QUESTION_REWRITE_SYS_TMPL, QUESTION_REWRITE_USER_TMPL]
)


class QAPair(BaseModel):
    """改写后的问答对数据模型。"""

    query: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The answer to the question.")


class State(TypedDict):
    """问答改写工作流的状态字典。"""

    document_id: str
    original_query: str
    original_answer: str
    query: str
    answer: str
    context: str


llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus").with_structured_output(
    QAPair
)


def generate(state: State):
    """调用 LLM 对原始问答对进行改写，返回改写后的问题和答案。"""
    prompt = prompt_template.invoke(
        {
            "context_str": state["context"],
            "question": state["original_query"],
            "answer": state["original_answer"],
        }
    )
    qa_pair = llm.invoke(prompt)
    return {
        "query": qa_pair.query,
        "answer": qa_pair.answer,
    }


async def main(input_fn: str, output_dir: str, max_concurrency: int = 8):
    """异步主函数：筛选已验证的问答对并批量改写，结果写入 qa_pairs.rewritten.json。"""
    graph_builder = StateGraph(State)
    graph_builder.add_node(generate)
    graph_builder.add_edge(START, "generate")
    graph = graph_builder.compile()

    with open(input_fn, "r") as f:
        data = json.load(f)

    inputs = []
    qa_pair_dict = {}
    logger.info(f"{len(data)} documents loaded from {input_fn}")
    for doc in data:
        # 跳过未通过验证（verdict=0）的问答对
        if not doc["metadatas"]["verdict"]:
            continue
        original_query = doc["query"]
        original_answer = doc["ground_truth"]["content"]
        doc_id = doc["metadatas"]["document_id"]
        inputs.append(
            {
                "document_id": doc_id,
                "original_query": original_query,
                "original_answer": original_answer,
                "context": doc["ground_truth"]["contexts"][0],
            }
        )
        qa_pair_dict[doc_id] = doc
    logger.info(f"{len(qa_pair_dict)} documents are valid and will be rewritten")

    outputs = await graph.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
    logger.info(f"{len(outputs)} documents generated")
    qa_pairs = []
    for result in outputs:
        doc_id = result["document_id"]
        doc = qa_pair_dict[doc_id]
        doc["query"] = result["query"]
        doc["ground_truth"]["content"] = result["answer"]
        doc["metadatas"]["original_query"] = result["original_query"]
        doc["metadatas"]["original_answer"] = result["original_answer"]
        qa_pairs.append(doc)

    output_path = Path(output_dir) / "qa_pairs.rewritten.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info(f"{len(qa_pairs)} documents rewritten to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fn", "-i", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.input_fn is None:
        raise RuntimeError("Missing required argument: --input_fn/-i")
    asyncio.run(
        main(
            input_fn=args.input_fn,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
        )
    )
