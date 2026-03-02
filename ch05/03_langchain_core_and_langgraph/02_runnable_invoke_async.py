"""Runnable 接口 - ainvoke（异步调用）

本脚本演示 LangChain 中 Runnable 接口的 ainvoke 方法。
ainvoke 是 invoke 的异步版本，适用于异步上下文（async/await）。
异步调用可以在 I/O 密集型场景（如网络请求）中避免阻塞事件循环，
从而提升并发性能。
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
    """主函数：创建异步 Runnable 并调用 ainvoke。"""
    runnable = RunnableGenerator(agen_regards)
    # ainvoke 方法：异步调用，等待所有 token 生成完毕后返回拼接结果
    response = await runnable.ainvoke(None)
    print(response)
    # 输出：
    # Have a nice day


if __name__ == "__main__":
    asyncio.run(main())
