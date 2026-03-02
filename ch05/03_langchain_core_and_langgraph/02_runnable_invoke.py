"""Runnable 接口 - invoke（同步调用）

本脚本演示 LangChain 中 Runnable 接口的 invoke 方法。
invoke 是最基础的调用方式：传入单个输入，返回完整的输出结果。
RunnableGenerator 可以将普通的 Python 生成器函数包装为 Runnable（可运行对象），
使其具备 invoke、batch、stream 等统一接口。
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


# 将生成器函数包装为 Runnable 对象，获得统一的调用接口
runnable = RunnableGenerator(gen_regards)
# invoke 方法：同步调用，将生成器的所有输出拼接为完整字符串返回
response = runnable.invoke(None)
print(response)
# 输出
# Have a nice day
