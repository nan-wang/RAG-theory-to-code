import argparse
import asyncio
import dotenv
import json
from argparse import BooleanOptionalAction
from loguru import logger

from prompts.keypoints_extract_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pathlib import Path
from datamodels import KeyPoints

dotenv.load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_docs",
        "-n",
        default=-1,
        type=int,
        help="The number of documents to be processed.",
    )
    parser.add_argument(
        "--ground-truth",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to extract keypoints from ground truth or response.",
    )
    parser.add_argument(
        "--response",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to extract keypoints from ground truth or response.",
    )
    parser.add_argument(
        "--max_concurrency", default=8, type=int, help="Max concurrent batch runs."
    )
    parser.add_argument("--input_fn", "-i", default=None, help="The input file path.")
    parser.add_argument(
        "--output_dir", "-o", default="./metrics", help="The output file path."
    )
    args = parser.parse_args()
    if args.input_fn is None:
        raise RuntimeError("Please provide the input file path via --input_fn/-i.")
    asyncio.run(
        _main(
            args.num_docs,
            args.output_dir,
            args.ground_truth,
            args.response,
            args.max_concurrency,
            args.input_fn,
        )
    )


async def _main(
    num_docs, output_dir, ground_truth, response, max_concurrency, input_fn
):
    KE_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

    KE_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

    prompt = ChatPromptTemplate.from_messages(messages=[KE_SYS_TMPL, KE_USER_TMPL])

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus").with_structured_output(
        KeyPoints
    )

    chain = prompt | llm

    with open(input_fn, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded from {input_fn}")

    docs = data[:num_docs]
    logger.info(
        f"Selected {num_docs if num_docs!=-1 else len(data)} from {len(data)} documents"
    )

    if ground_truth:
        gt_inputs = [
            {"question": doc["query"], "answer": doc["ground_truth"]["content"]}
            for doc in docs
        ]
        gt_outputs = await chain.abatch(
            gt_inputs, config=RunnableConfig(max_concurrency=max_concurrency)
        )
        for doc, result in zip(docs, gt_outputs):
            try:
                doc["ground_truth"]["keypoints"] = result.keypoints
            except Exception as e:
                logger.info(f"Error: {e}")
                logger.info(
                    f"Failed to extract keypoints from ground truth for result: {result}"
                )
                doc["_gt_failed"] = True

    if response:
        rsp_inputs = [
            {"question": doc["query"], "answer": doc["response"]["content"]}
            for doc in docs
        ]
        rsp_outputs = await chain.abatch(
            rsp_inputs, config=RunnableConfig(max_concurrency=max_concurrency)
        )
        for doc, result in zip(docs, rsp_outputs):
            try:
                doc["response"]["keypoints"] = result.keypoints
            except Exception as e:
                logger.info(f"Error: {e}")
                logger.info(
                    f"Failed to extract keypoints from response for result: {result}"
                )

    results = [doc for doc in docs if not doc.get("_gt_failed")]
    # Clean up temporary marker
    for doc in results:
        doc.pop("_gt_failed", None)

    output_fn = Path(output_dir) / "keypoints.json"
    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fn, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved the results to {output_fn}")


if __name__ == "__main__":
    main()
