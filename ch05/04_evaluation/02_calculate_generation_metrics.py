import json
import asyncio

import click
from pathlib import Path
import dotenv
import re

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from prompts.keypoints_verify_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain import QAWithSourcesChain

from datamodels import KeyPoint


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
    '--loyalty/--no-loyalty',
    default=False
)
@click.option(
    '--hallucination/--no-hallucination',
    default=False
)
@click.option(
    '--noise-sensitivity/--no-noise-sensitivity',
    default=False
)
@click.option(
    '--context-utility-ratio/--no-context-utility-ratio',
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
def main(num_docs, output_path, loyalty, hallucination, noise_sensitivity, context_utility_ratio, max_concurrency, input_fn):
    asyncio.run(_main(num_docs, output_path, loyalty, hallucination, noise_sensitivity, context_utility_ratio, max_concurrency, input_fn))


async def _main(num_docs, output_path, loyalty, hallucination, noise_sensitivity, context_utility_ratio, max_concurrency, input_fn):
    with open(input_fn) as f:
        docs = json.load(f)
        response_loyalty_kp = []
        response_hallucination_kp = []
        response_noise_sensitivity_kp = []
        response_context_utility_ratio_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            answer = doc["ground_truth"]["content"]
            response = doc["response"]["content"]
            context = doc["response"]["contexts"][0]
            for k in doc["response"]["keypoints"]:
                response_loyalty_kp.append(
                    KeyPoint(question=question, answer=context, keypoint=k))
                response_hallucination_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )
                response_noise_sensitivity_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )
            for k in doc["ground_truth"]["keypoints"]:
                response_context_utility_ratio_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )

    KV_SYS_TMPL = (
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

    KV_USER_TMPL = (
        HumanMessagePromptTemplate.from_template(USER_PROMPT))

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            KV_SYS_TMPL,
            KV_USER_TMPL
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    match = re.compile(r'\[\[\[([^\]]+)\]\]\]')

    chain = (prompt | llm | StrOutputParser())
    config = RunnableConfig(max_concurrency=max_concurrency)

    if loyalty:
        inputs = [{"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint} for kp in response_loyalty_kp]
        outputs = await chain.abatch(inputs, config=config)
        for kp, result in zip(response_loyalty_kp, outputs):
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")

        response_loyalty_list = response_loyalty_kp
        output_fn = Path(output_path) / "metrics" / "generation_loyalty.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([kp.dict() for kp in response_loyalty_list], f, indent=4, ensure_ascii=False)
        supported_kp = sum([1 for kp in response_loyalty_list if kp.label == "Relevant"])
        response_loyalty = supported_kp/len(response_loyalty_list)
        print(f"response_loyalty ↑: {response_loyalty:.3f}")

    if hallucination:
        # Flatten all keypoints from all groups into a single list
        flat_kps = []
        flat_indices = []  # (group_idx, position_in_group)
        for group_idx, kp_group in enumerate(response_hallucination_kp):
            for pos, kp in enumerate(kp_group):
                flat_kps.append(kp)
                flat_indices.append((group_idx, pos))

        inputs = [{"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint} for kp in flat_kps]
        outputs = await chain.abatch(inputs, config=config)

        for kp, result in zip(flat_kps, outputs):
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")

        # Reconstruct: a group is hallucination=True unless any kp is "Relevant"
        result_list = []
        for group_idx, kp_group in enumerate(response_hallucination_kp):
            label = True
            for kp in kp_group:
                if kp.label == "Relevant":
                    label = False
            result_list.append((kp_group, label))

        output_fn = Path(output_path) / "metrics" / "generation_hallucination.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([{"details": [kp.dict() for kp in kp_g], "is_hallucination": l} for kp_g, l in result_list], f, indent=4, ensure_ascii=False)
        hallucination_kp = sum([label for kp_group, label in result_list])
        hallucination_score = hallucination_kp/len(result_list)
        print(f"hallucination score ↓: {hallucination_score:.3f}")

    if noise_sensitivity:
        # Flatten all (claim_context, claim_ans) pairs
        flat_kps = []
        flat_indices = []  # (pair_idx, 0=context/1=ans)
        for pair_idx, (claim_context, claim_ans) in enumerate(response_noise_sensitivity_kp):
            flat_kps.append(claim_context)
            flat_indices.append((pair_idx, 0))
            flat_kps.append(claim_ans)
            flat_indices.append((pair_idx, 1))

        inputs = [{"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint} for kp in flat_kps]
        outputs = await chain.abatch(inputs, config=config)

        for kp, result in zip(flat_kps, outputs):
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")

        # Reconstruct with original logic
        result_list = []
        for claim_context, claim_ans in response_noise_sensitivity_kp:
            label = False
            if claim_context.label == "Relevant":
                label = True
            if claim_ans.label == "Relevant":
                label = False
            result_list.append(((claim_context, claim_ans), label))

        output_fn = Path(output_path) / "metrics" / "generation_noise_sensitivity.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([
                {
                    "details": [kp.dict() for kp in kp_g],
                    "is_noise": l
                } for kp_g, l in result_list
            ], f, indent=4, ensure_ascii=False)
        noise_sensitivity_kp = sum([label for kp_group, label in result_list])
        noise_sensitivity_score = noise_sensitivity_kp/len(result_list)
        print(f"noise sensitivity score ↓: {noise_sensitivity_score:.3f} ({noise_sensitivity_kp}/{len(result_list)})")

    if context_utility_ratio:
        # Flatten all (claim_context, claim_ans) pairs
        flat_kps = []
        for claim_context, claim_ans in response_context_utility_ratio_kp:
            flat_kps.append(claim_context)
            flat_kps.append(claim_ans)

        inputs = [{"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint} for kp in flat_kps]
        outputs = await chain.abatch(inputs, config=config)

        for kp, result in zip(flat_kps, outputs):
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")

        # Reconstruct with original logic
        result_list = []
        for claim_context, claim_ans in response_context_utility_ratio_kp:
            supported_by_cxt = True
            supported_by_cxt_and_ans = False
            if claim_context.label != "Relevant":
                supported_by_cxt = False
            elif claim_context.label == "Relevant":
                supported_by_cxt_and_ans = True
            if claim_ans.label != "Relevant":
                supported_by_cxt_and_ans = False
            result_list.append(((claim_context, claim_ans), supported_by_cxt, supported_by_cxt_and_ans))

        output_fn = Path(output_path) / "metrics" / "generation_context_utility_ratio.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([{
                "claim_cxt": kp_cxt.dict(),
                "claim_ans": kp_ans.dict(),
                "supported_by_cxt": l_cxt,
                "supported_by_cxt_and_ans": l_ans
            } for (kp_cxt, kp_ans), l_cxt, l_ans in result_list], f, indent=4, ensure_ascii=False)
        context_utility_ratio_den = sum([l_cxt for (_, _), l_cxt, _ in result_list])
        context_utility_ratio_num = sum([l_ans for (_, _), _, l_ans in result_list])
        context_utility_ratio_score = context_utility_ratio_num / context_utility_ratio_den if context_utility_ratio_den else 0
        print(f"context utility ratio score ↑: {context_utility_ratio_score:.3f} ({context_utility_ratio_num}/{context_utility_ratio_den})")


if __name__ == '__main__':
    # python calculate_generation_metrics.py -n 100 -o data_metrics/v20241219/ch0503_naive --loyalty --hallucination --noise-sensitivity --context-utility-ratio data_metrics/v20241219/ch0503_naive/keypoints.json
    main()
