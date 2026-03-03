import json
import asyncio
import click

from pathlib import Path
import dotenv
from utils import dump_metrics, dump_scores, verify_keypoints
from datamodels import KeyPoint

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from prompts.groundtruth_keypoints_verify_prompt import SYSTEM_PROMPT as GT_SYSTEM_PROMPT, USER_PROMPT as GT_USER_PROMPT
from prompts.answer_keypoints_verify_prompt import SYSTEM_PROMPT as ANS_SYSTEM_PROMPT, USER_PROMPT as ANS_USER_PROMPT
from langchain_core.output_parsers import StrOutputParser


dotenv.load_dotenv()


@click.command()
@click.option(
    '--num_docs',
    '-n',
    default=-1,
    help='The number of documents to be processed.')
@click.option(
    '--output_path',
    '-o',
    default="./metrics",
    help='The output file path.',
    type=click.Path(file_okay=False, dir_okay=True, writable=True)
)
@click.option(
    '--precision/--no-precision',
    default=True
)
@click.option(
    '--recall/--no-recall',
    default=True
)
@click.option(
    '--max_concurrency',
    default=8,
    type=int,
    help='Max concurrent batch runs.'
)
@click.argument(
    'input_fn',
    default="keypoints.json")
def main(num_docs, output_path, precision, recall, max_concurrency, input_fn):
    asyncio.run(_main(num_docs, output_path, precision, recall, max_concurrency, input_fn))


async def _main(num_docs, output_path, precision, recall, max_concurrency, input_fn):
    with open(input_fn, "r") as f:
        docs = json.load(f)
        rsp_kp = []
        ans_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            answer = doc["ground_truth"]["content"]
            response = doc["response"]["content"]
            for k in doc["response"]["keypoints"]:
                rsp_kp.append(
                    KeyPoint(question=question, answer=answer, keypoint=k))
            for k in doc["ground_truth"]["keypoints"]:
                ans_kp.append(
                    KeyPoint(question=question, answer=response, keypoint=k))

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus")

    # chain for ground-truth verification (used by recall)
    gt_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(GT_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(GT_USER_PROMPT),
    ])
    chain_gt = (gt_prompt | llm | StrOutputParser())

    # chain for answer keypoints verification (used by precision)
    ans_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ANS_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(ANS_USER_PROMPT),
    ])
    chain_ans = (ans_prompt | llm | StrOutputParser())

    # calculate the precision
    if precision:
        precision_list = await verify_keypoints(rsp_kp, chain_ans, max_concurrency)
        dump_metrics(
            precision_list,
            Path(output_path) / "metrics" / "global_precision.json")
        supported_kp = sum([1 for kp in precision_list if kp.label == "Relevant"])
        precision_score = supported_kp / len(precision_list)
        print(f"precision: {precision_score:.3f}")

    if recall:
        recall_list = await verify_keypoints(ans_kp, chain_gt, max_concurrency)
        dump_metrics(
            recall_list,
            Path(output_path) / "metrics" / "global_recall.json")
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
    dump_scores(scores, Path(output_path) / "metrics" / "scores.json")


if __name__ == '__main__':
    # python calculate_global_metrics.py -n 100 -o data_metrics/v20241219/ch0503_naive/ --precision --recall data_metrics/v20241219/ch0503_naive/keypoints.json
    main()
