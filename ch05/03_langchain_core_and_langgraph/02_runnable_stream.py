"""Runnable 接口 - stream（同步流式输出）

本脚本演示 LangChain 中 Runnable 接口的 stream 方法。
stream 方法返回一个迭代器，每次迭代产出一个 token（或数据块），
适用于需要实时展示生成进度的场景（如聊天界面逐字输出）。
与 invoke 不同，stream 不会等待所有内容生成完毕才返回。
"""

from langchain_core.runnables import RunnableGenerator


def gen_regards(input):
    """逐 token 生成问候语的生成器函数。

    Args:
        input: Runnable 接口要求的输入参数，此处未使用。

    Yields:
        str: 问候语的每个 token 片段。
    """
    for token in ["Have", " a", " nice", " day"]:
        yield token


runnable = RunnableGenerator(gen_regards)
# stream 方法：返回迭代器，逐个产出 token，实现流式输出
for c in runnable.stream(None):
    print(f"{c}|")
# 输出:
# Have|
# a|
# nice|
# day|
