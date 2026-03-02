"""Runnable 接口 - astream（异步流式输出）

本脚本演示 LangChain 中 Runnable 接口的 astream 方法。
astream 是 stream 的异步版本，返回异步迭代器，
可以在异步上下文中逐个产出 token，不阻塞事件循环。
适用于异步 Web 框架（如 FastAPI）中的流式响应场景。
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
    """主函数：创建异步 Runnable 并通过 astream 流式输出。"""
    runnable = RunnableGenerator(agen_regards)

    # astream 方法：返回异步迭代器，使用 async for 逐个获取 token
    async for c in runnable.astream(None):
        print(f"{c}|")
    # 输出：
    # Have |
    # a |
    # nice |
    # day |


if __name__ == "__main__":
    asyncio.run(main())
