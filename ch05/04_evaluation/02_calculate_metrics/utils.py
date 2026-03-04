"""指标计算阶段的通用工具函数：上下文拆分、结果保存和要点批量验证。"""

import re
from pathlib import Path
import json

from langchain_core.runnables import RunnableConfig


def split_contexts(contexts):
    """按 article_title: 标记将拼接上下文字符串拆分为独立块列表。"""
    pattern = r"(?=article_title:)"
    # Split the text using the regex pattern
    return [part.strip() for part in re.split(pattern, contexts) if part.strip()]


def dump_metrics(results, output_fn):
    """将 KeyPoint 列表序列化为 JSON 文件。"""
    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fn, "w") as f:
        json.dump([kp.dict() for kp in results], f, indent=4, ensure_ascii=False)
    print(f"Dumping the results to {output_fn}")


def dump_scores(scores: dict, output_fn):
    """Read existing scores.json (if any), update with new scores, and write back."""
    path = Path(output_fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
    existing.update(scores)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Dumping the scores to {output_fn}")


async def verify_keypoints(keypoints, lc_chain, max_concurrency=8):
    """批量调用 LangChain 链验证要点标签，返回标注后的 KeyPoint 列表。"""
    # 正则表达式提取 LLM 输出中的 [[[标签]]] 结论
    match = re.compile(r"\[\[\[([^\]]+)\]\]\]")
    inputs = [
        {"question": kp.question, "answer": kp.answer, "keypoint": kp.keypoint}
        for kp in keypoints
    ]
    outputs = await lc_chain.abatch(
        inputs, config=RunnableConfig(max_concurrency=max_concurrency)
    )
    for kp, result in zip(keypoints, outputs):
        rsp = match.search(result)
        if rsp:
            kp.label = rsp.group(1)
        else:
            print(f"Failed to extract the label for the keypoint: {result}")
    return list(keypoints)
