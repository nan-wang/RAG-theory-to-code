"""Runnable 接口 - abatch（异步批量调用）

本脚本演示 LangChain 中 Runnable 接口的 abatch 方法。
abatch 是 batch 的异步版本，可以并发处理多个输入，
在 I/O 密集型场景中能显著提升吞吐量。
"""

import asyncio

from langchain_core.runnables import RunnableGenerator


async def agen_regards(input):
    """异步逐 token 生成问候语的异步生成器函数。

    Args:
        input: Runnable 接口要求的输入参数，此处未使用。

    Yields:
        str: 问候语的每个 token 片段。
    """
    for token in ["Have", " a", " nice", " day"]:
        yield token


async def main():
    """主函数：创建异步 Runnable 并调用 abatch。"""
    runnable = RunnableGenerator(agen_regards)
    # abatch 方法：异步批量调用，并发处理输入列表中的每个元素
    response = await runnable.abatch([None, None])
    print(response)
    # 输出：
    # ['Have a nice day', 'Have a nice day']


if __name__ == "__main__":
    asyncio.run(main())
