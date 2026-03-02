"""LangGraph Send 机制 - 动态并行创建节点实例

本脚本演示 LangGraph 中 Send 机制的用法。
Send 允许在条件边中动态创建多个节点实例并行执行，
每个实例可以接收不同的输入。这类似于 map-reduce 模式中的 map 阶段。

典型应用场景：对列表中的每个元素并行执行相同的处理逻辑。
"""

import operator
from typing import Annotated

from langgraph.graph import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict


class OverallState(TypedDict):
    """全局状态定义。

    Attributes:
        subjects: 主题列表，每个主题将生成一个笑话。
        jokes: 生成的笑话列表，使用 add 归约器收集各并行节点的结果。
    """
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


def continue_to_jokes(state: OverallState):
    """路由函数：为每个主题创建一个 Send 对象，实现动态并行。

    Send(node_name, input_dict) 表示创建一个节点实例：
    - node_name: 要执行的目标节点
    - input_dict: 传给该节点实例的独立输入

    Args:
        state: 当前全局状态。

    Returns:
        list[Send]: Send 对象列表，每个对应一个并行执行的节点实例。
    """
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


graph_builder = StateGraph(OverallState)
# 使用 lambda 定义节点逻辑：根据传入的 subject 生成笑话
graph_builder.add_node(
    "generate_joke", lambda state: {"jokes": [f"讲一个关于{state['subject']}的冷笑话"]}
)
# 从 START 出发通过条件边调用 continue_to_jokes，动态创建多个并行节点
graph_builder.add_conditional_edges(START, continue_to_jokes)
graph_builder.add_edge("generate_joke", END)
app = graph_builder.compile()

# 传入两个主题，将并行生成两个笑话
output = app.invoke({"subjects": ["汽车", "小猫"]})
print(output)
