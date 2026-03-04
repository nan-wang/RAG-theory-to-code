"""计算 RAG 系统的全局精确率、召回率和 F1 指标。"""

import argparse
import json
import asyncio

from argparse import BooleanOptionalAction
from pathlib import Path
import dotenv
from utils import dump_metrics, dump_scores, verify_keypoints
from datamodels import KeyPoint

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from prompts.groundtruth_keypoints_verify_prompt import (
    SYSTEM_PROMPT as GT_SYSTEM_PROMPT,
    USER_PROMPT as GT_USER_PROMPT,
)
from prompts.answer_keypoints_verify_prompt import (
    SYSTEM_PROMPT as ANS_SYSTEM_PROMPT,
    USER_PROMPT as ANS_USER_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()


def main():
    """解析命令行参数并启动异步全局指标计算流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fn", "-i", default=None, help="The input keypoints JSON file."
    )
    parser.add_argument(
        "--num_docs",
        "-n",
        default=-1,
        type=int,
        help="The number of documents to be processed.",
    )
    parser.add_argument(
        "--output_dir", "-o", default="./metrics", help="The output directory."
    )
    parser.add_argument("--precision", action=BooleanOptionalAction, default=True)
    parser.add_argument("--recall", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max_concurrency", default=8, type=int, help="Max concurrent batch runs."
    )
    args = parser.parse_args()
    if args.input_fn is None:
        raise RuntimeError("Missing required argument: --input_fn/-i")
    asyncio.run(
        _main(
            args.num_docs,
            args.output_dir,
            args.precision,
            args.recall,
            args.max_concurrency,
            args.input_fn,
        )
    )


async def _main(num_docs, output_dir, precision, recall, max_concurrency, input_fn):
    """异步计算全局精确率和召回率，并将结果写入文件。"""
    with open(input_fn, "r") as f:
        docs = json.load(f)
        rsp_kp = []
        ans_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            answer = doc["ground_truth"]["content"]
            response = doc["response"]["content"]
            for k in doc["response"]["keypoints"]:
                rsp_kp.append(KeyPoint(question=question, answer=answer, keypoint=k))
            for k in doc["ground_truth"]["keypoints"]:
                ans_kp.append(KeyPoint(question=question, answer=response, keypoint=k))

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus")

    # chain for ground-truth verification (used by recall)
    gt_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(GT_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(GT_USER_PROMPT),
        ]
    )
    chain_gt = gt_prompt | llm | StrOutputParser()

    # chain for answer keypoints verification (used by precision)
    ans_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(ANS_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(ANS_USER_PROMPT),
        ]
    )
    chain_ans = ans_prompt | llm | StrOutputParser()

    # calculate the precision
    if precision:
        precision_list = await verify_keypoints(rsp_kp, chain_ans, max_concurrency)
        dump_metrics(
            precision_list, Path(output_dir) / "metrics" / "global_precision.json"
        )
        supported_kp = sum([1 for kp in precision_list if kp.label == "Relevant"])
        precision_score = supported_kp / len(precision_list)
        print(f"precision: {precision_score:.3f}")

    if recall:
        recall_list = await verify_keypoints(ans_kp, chain_gt, max_concurrency)
        dump_metrics(recall_list, Path(output_dir) / "metrics" / "global_recall.json")
        supported_kp = sum([1 for kp in recall_list if kp.label == "Relevant"])
        recall_score = supported_kp / len(recall_list)
        print(f"recall: {recall_score:.3f}")

    if precision and recall:
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score)
        print(f"f1: {f1:.3f}")

    scores = {}
    if precision:
        scores["precision"] = precision_score
    if recall:
        scores["recall"] = recall_score
    if precision and recall:
        scores["f1"] = f1
    dump_scores(scores, Path(output_dir) / "metrics" / "scores.json")


if __name__ == "__main__":
    main()
