import json
import asyncio
from pathlib import Path
import dotenv
import click
import re

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from prompts.groundtruth_keypoints_verify_prompt import SYSTEM_PROMPT as GT_SYSTEM_PROMPT, USER_PROMPT as GT_USER_PROMPT
from prompts.answer_keypoints_verify_prompt import SYSTEM_PROMPT as ANS_SYSTEM_PROMPT, USER_PROMPT as ANS_USER_PROMPT
from langchain_core.output_parsers import StrOutputParser

from datamodels import KeyPoint
from utils import dump_metrics, verify_keypoints


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
    default=False
)
@click.option(
    '--recall/--no-recall',
    default=False
)
@click.option(
    '--max_concurrency',
    default=8,
    type=int,
    help='Max concurrent batch runs.'
)
@click.argument(
    'input_fn',
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
def main(num_docs, output_path, precision, recall, max_concurrency, input_fn):
    asyncio.run(_main(num_docs, output_path, precision, recall, max_concurrency, input_fn))


async def _main(num_docs, output_path, precision, recall, max_concurrency, input_fn):
    with open(input_fn) as f:
        docs = json.load(f)
        cxt_precision_kp = []
        cxt_recall_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            context = doc["response"]["contexts"][0]
            for k in doc["ground_truth"]["keypoints"]:
                cxt_recall_kp.append(
                    KeyPoint(question=question, answer=context, keypoint=k))
            for ctx in context.split("\n"):
                ctx = ctx.strip("\n")
                if not ctx:
                    continue
                cur_context_kp = []
                for k in doc["response"]["keypoints"]:
                    cur_context_kp.append(
                        KeyPoint(question=question, answer=ctx, keypoint=k))
                cxt_precision_kp.append(cur_context_kp)

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus")
    match = re.compile(r'\[\[\[([^\]]+)\]\]\]')

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

    config = RunnableConfig(max_concurrency=max_concurrency)

    if precision:
        # Flatten all keypoints from all groups into a single list
        flat_kps = []
        group_sizes = []
        for kp_group in cxt_precision_kp:
            group_sizes.append(len(kp_group))
            flat_kps.extend(kp_group)

        inputs = [{"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint} for kp in flat_kps]
        outputs = await chain_ans.abatch(inputs, config=config)

        for kp, result in zip(flat_kps, outputs):
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")

        # Reconstruct groups and determine label per group
        context_precision_list = []
        offset = 0
        for kp_group, size in zip(cxt_precision_kp, group_sizes):
            label = False
            for kp in kp_group:
                if kp.label == "Relevant":
                    label = True
            context_precision_list.append((kp_group, label))
            offset += size

        output_fn = Path(output_path) / "metrics" / "retrieval_context_precision.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([([kp.dict() for kp in kp_g], l) for kp_g, l in context_precision_list], f, indent=4, ensure_ascii=False)
        supported_kp = sum([label for kp_group, label in context_precision_list])
        precision_score = supported_kp/len(context_precision_list)
        print(f"context_precision: {precision_score:.3f}")

    if recall:
        context_recall_list = await verify_keypoints(cxt_recall_kp, chain_gt, max_concurrency)

        output_fn = Path(output_path) / "metrics" / "retrieval_keypoints_recall.json"
        dump_metrics(context_recall_list, output_fn)
        supported_kp = sum([1 for kp in context_recall_list if kp.label == "Relevant"])
        keypoints_recall = supported_kp/len(context_recall_list)
        print(f"context_recall: {keypoints_recall:.3f}")


if __name__ == '__main__':
    # python calculate_retrieval_metrics.py -n 100 -o data_metrics/v20241219/ch0503_naive/metrics --precision data_metrics/v20241219/ch0503_naive/keypoints.json
    # python calculate_retrieval_metrics.py -n 100 -o data_metrics/v20241219/ch0503_naive/metrics --precision --recall data_metrics/v20241219/ch0503_naive/keypoints.json
    main()
