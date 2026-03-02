"""LangGraph 状态归约器 - 使用 Annotated 实现列表追加

本脚本演示 LangGraph 中 Annotated 归约器（Reducer）的用法。
默认情况下，节点返回的值会覆盖状态中的同名字段。
通过 Annotated[type, reducer_fn] 可以指定一个归约函数，
使新值与旧值合并而非覆盖。

这里使用 operator.add 作为归约器，实现列表的自动追加。
"""

from operator import add
from typing import Annotated

from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict


class State(TypedDict):
    """图的状态定义。

    Attributes:
        msg_list: 消息列表，使用 add 归约器。
            Annotated[list[str], add] 表示节点返回的列表会与已有列表拼接，
            而非覆盖。等效于 old_list + new_list。
    """
    msg_list: Annotated[list[str], add]


def node_1(state: State):
    """节点1：返回包含 "hello," 的列表，会被追加到 msg_list。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 要追加到 msg_list 的新元素。
    """
    return {"msg_list": ["hello,"]}


def node_2(state: State):
    """节点2：返回包含 "world!" 的列表，会被追加到 msg_list。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 要追加到 msg_list 的新元素。
    """
    return {"msg_list": ["world!"]}


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
app = graph_builder.compile()
# 初始状态为空列表，经过两个节点后变为 ["hello,", "world!"]
output = app.invoke({"msg_list": []})
print(output)
