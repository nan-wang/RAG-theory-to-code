"""对合成问答对的质量进行验证，过滤含糊或依赖外部信息的问题。"""

import asyncio
import argparse
import json
from pathlib import Path
from loguru import logger

import dotenv
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from prompts.validate_question_answer_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv(".env")

QUESTION_VALIDATE_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
QUESTION_VALIDATE_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[QUESTION_VALIDATE_SYS_TMPL, QUESTION_VALIDATE_USER_TMPL]
)


class QAFeedback(BaseModel):
    """LLM 对问题质量的评估结果，包含反馈说明和 0/1 裁决分数。"""

    feedback: str = Field(..., description="Feedback for the question.")
    verdict: int = Field(..., description="Score for the question.")


llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus").with_structured_output(
    QAFeedback
)


class State(TypedDict):
    """问答验证工作流的状态字典。"""

    context: str
    query: str
    answer: str
    feedback: str
    response: str
    document_id: str
    verdict: int


def verify(state: State):
    """调用 LLM 对问答对进行质量验证，返回反馈和裁决分数。"""
    prompt = prompt_template.invoke(
        {
            "context_str": state["context"],
            "question": state["query"],
            "answer": state["answer"],
        }
    )
    response = llm.invoke(prompt)
    return {
        "feedback": response.feedback,
        "verdict": response.verdict,
    }


async def main(input_fn: str, output_dir: str, max_concurrency: int = 8):
    """异步主函数：批量验证问答对质量，结果写入 qa_pairs.validated.json。"""
    graph_builder = StateGraph(State)
    graph_builder.add_node(verify)
    graph_builder.add_edge(START, "verify")
    graph_builder.add_edge("verify", END)
    graph = graph_builder.compile()

    with open(input_fn, "r") as f:
        data = json.load(f)
    logger.info(f"Read {len(data)} questions from {input_fn}")

    inputs = [
        {
            "query": doc["query"],
            "answer": doc["ground_truth"]["content"],
            "context": doc["ground_truth"]["contexts"][0],
            "document_id": doc["metadatas"]["document_id"],
        }
        for doc in data
    ]
    outputs = await graph.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
    # 用 document_id 作为键，便于后续将验证结果回填到原始文档
    result_dict = {}
    for result in outputs:
        result_dict[result["document_id"]] = result
    results = []
    for doc in data:
        document_id = doc["metadatas"]["document_id"]
        output_doc = doc
        output_doc["metadatas"]["feedback"] = result_dict[document_id]["feedback"]
        output_doc["metadatas"]["verdict"] = result_dict[document_id]["verdict"]
        results.append(output_doc)
    output_path = Path(output_dir) / "qa_pairs.validated.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(results)} questions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fn", "-i", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.input_fn, args.output_dir, args.max_concurrency))
