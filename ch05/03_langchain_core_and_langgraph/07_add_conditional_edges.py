"""LangGraph 条件边 - 根据状态动态选择下一个节点

本脚本演示 LangGraph 中条件边（Conditional Edge）的用法。
与普通边（固定连接两个节点）不同，条件边通过一个路由函数
在运行时根据当前状态动态决定下一个要执行的节点。
这是实现分支逻辑（if/else）的关键机制。
"""

from langgraph.graph import StateGraph, START

from typing_extensions import TypedDict


class State(TypedDict):
    """图的状态定义。

    Attributes:
        input: 输入值，用于路由判断。
        msg: 记录执行路径的消息字符串。
    """
    input: int
    msg: str


def node_1(state: State):
    """节点1：在消息中追加自身名称。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 更新后的 msg 字段。
    """
    return {"msg": state.get("msg", "") + "node_1"}


def node_2(state: State):
    """节点2：在消息中追加自身名称。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 更新后的 msg 字段。
    """
    return {"msg": state.get("msg", "") + "node_2"}


def routing_function(state: State):
    """路由函数：根据 input 值决定下一个执行的节点。

    条件边的核心逻辑：返回目标节点的名称字符串。

    Args:
        state: 当前图的状态。

    Returns:
        str: 下一个节点的名称（"node_1" 或 "node_2"）。
    """
    # 通过输入的input值来决定下一个节点
    if state["input"] >= 0:
        return "node_1"
    else:
        return "node_2"


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
# add_conditional_edges：从 START 出发，由 routing_function 决定下一个节点
graph_builder.add_conditional_edges(START, routing_function)

app = graph_builder.compile()
print(app.invoke({"input": 1}))
# 输出：{'input': 1, 'msg': 'node_1'}
print(app.invoke({"input": -1}))
# 输出：{'input': -1, 'msg': 'node_2'}
