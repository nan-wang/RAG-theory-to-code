"""Runnable 接口 - batch（同步批量调用）

本脚本演示 LangChain 中 Runnable 接口的 batch 方法。
batch 方法接收一个输入列表，对每个输入分别调用 Runnable，
返回一个与输入列表等长的结果列表。
适用于需要对多个输入进行相同处理的场景。
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
# batch 方法：传入输入列表，对每个输入分别执行，返回结果列表
response = runnable.batch([None, None])
print(response)
# 输出: ['Have a nice day', 'Have a nice day']
