"""利用 LLM 从向量索引中随机采样文档并生成问答三元组的数据合成脚本。"""

import asyncio
import argparse
import json
import random
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from synthetic_data_prompt import SYSTEM_PROMPT, USER_PROMPT
from loguru import logger

dotenv.load_dotenv()


class QATriplet(BaseModel):
    """Structured output for a single QA triplet used in embedding fine-tuning."""

    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The correct answer to the question.")
    negative_document: str = Field(
        ..., description="The wrong context not related to the question."
    )


class State(TypedDict):
    """LangGraph state shared across all nodes in the generation graph."""

    length: int
    clarity: str
    difficulty: str
    document_id: str
    context: str
    query: str
    answer: str
    negative_document: str


def select_length(_state: State):
    """Randomly sample a question-length target (in characters)."""
    return {"length": random.choice([8, 16, 32])}


def select_clarity(_state: State):
    """Randomly sample a clarity level for the generated question."""
    return {"clarity": random.choice(["简单", "基础", "困难"])}


def select_difficulty(_state: State):
    """Randomly sample an education-level difficulty for the generated question."""
    return {"difficulty": random.choice(["小学", "初中", "高中", "大学", "研究生博士"])}


def generate(state: State):
    """Invoke the LLM to produce a question, answer, and negative document from the context."""
    prompt = prompt_template.invoke(
        {
            "context_str": state["context"],
            "length": state["length"],
            "clarity": state["clarity"],
            "difficulty": state["difficulty"],
        }
    )
    qa_triplet = llm.invoke(prompt)
    return {
        "query": qa_triplet.question,
        "answer": qa_triplet.answer,
        "negative_document": qa_triplet.negative_document,
    }


QUESTION_GEN_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

QUESTION_GEN_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL]
)

llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus").with_structured_output(
    QATriplet
)


async def main(
    num_docs: int,
    index_dir: str,
    collection_name: str,
    output_dir: str,
    max_concurrency: int = 8,
):
    """Build the generation graph, run it against selected documents, and write outputs.

    Args:
        num_docs: Number of documents to process. -1 means all documents in the collection.
        index_dir: Path to the persisted Chroma index directory.
        collection_name: Name of the Chroma collection to query.
        output_dir: Directory where qa_pairs.json and qa_triplets.json are written.
        max_concurrency: Maximum number of LLM calls to run in parallel via abatch.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node(select_length)
    graph_builder.add_node(select_clarity)
    graph_builder.add_node(select_difficulty)
    graph_builder.add_node(generate)

    graph_builder.add_edge(START, "select_length")
    graph_builder.add_edge(START, "select_clarity")
    graph_builder.add_edge(START, "select_difficulty")
    graph_builder.add_edge("select_length", "generate")
    graph_builder.add_edge("select_clarity", "generate")
    graph_builder.add_edge("select_difficulty", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    vectorstore = Chroma(persist_directory=index_dir, collection_name=collection_name)
    ids = vectorstore.get()["ids"]
    logger.info(f"Total number of documents: {len(ids)}")
    random.shuffle(ids)

    if num_docs == -1:
        num_docs = len(ids)
    else:
        num_docs = min(num_docs, len(ids))

    logger.info(f"Generating {num_docs} documents...")
    selected_docs = {
        k: v
        for k, v in vectorstore.get(ids=ids[:num_docs]).items()
        if k in ("ids", "metadatas", "documents")
    }
    selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

    inputs = [
        {
            "document_id": doc["ids"],
            "context": doc["documents"],
        }
        for doc in selected_docs
    ]

    logger.info(f"Total number of selected documents: {len(inputs)}")
    outputs = await graph.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
    logger.info("Complete generation")

    qa_pairs = []
    qa_triplets = []
    for result in outputs:
        qa_doc = {
            "query": result["query"],
            "ground_truth": {
                "contexts": [result["context"]],
                "content": result["answer"],
            },
            "metadatas": {
                "length": result["length"],
                "clarity": result["clarity"],
                "difficulty": result["difficulty"],
                "document_id": result["document_id"],
            },
        }
        qa_triplet = {
            "anchor": result["query"],
            "positive": result["context"],
            "negative": result["negative_document"],
        }
        qa_pairs.append(qa_doc)
        qa_triplets.append(qa_triplet)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs_file = output_path / "qa_pairs.json"
    with open(pairs_file, "w") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(qa_pairs)} pairs to {pairs_file}")

    triplets_file = output_path / "qa_triplets.json"
    with open(triplets_file, "w") as f:
        json.dump(qa_triplets, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(qa_triplets)} triplets to {triplets_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QA triplets for embedding fine-tuning."
    )
    parser.add_argument(
        "--index_dir",
        default=None,
        required=False,
        help="Path to the Chroma index directory.",
    )
    parser.add_argument(
        "--num_docs",
        "-n",
        type=int,
        default=-1,
        help="Number of documents to process. -1 means all documents.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="./data_finetuning",
        help="Output directory for qa_pairs.json and qa_triplets.json.",
    )
    parser.add_argument(
        "--collection_name", default="olympic_games", help="Chroma collection name."
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent LLM calls.",
    )
    args = parser.parse_args()

    if args.index_dir is None:
        parser.error(
            "--index_dir is required. Please provide the path to the Chroma index directory."
        )

    asyncio.run(
        main(
            num_docs=args.num_docs,
            index_dir=args.index_dir,
            collection_name=args.collection_name,
            output_dir=args.output_dir,
            max_concurrency=args.max_concurrency,
        )
    )
