"""LangGraph 检查点机制 - 使用 MemorySaver 保存状态历史

本脚本演示 LangGraph 中检查点（Checkpoint）机制的用法。
MemorySaver 是一个内存中的检查点保存器，它会在每个节点执行后
自动保存图的完整状态快照。

通过 thread_id 可以区分不同的对话/执行线程，
通过 get_state_history 可以回溯整个执行过程中的状态变化。
"""

from operator import add
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    """图的状态定义。

    Attributes:
        input: 输入字符串。
        node_trace: 节点执行轨迹，使用 add 归约器记录经过的节点。
    """
    input: str
    node_trace: Annotated[list[str], add]


def node_a(state: State):
    """节点A：在执行轨迹中记录 "a"。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 要追加到 node_trace 的记录。
    """
    return {"node_trace": ["a"]}


def node_b(state: State):
    """节点B：在执行轨迹中记录 "b"。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 要追加到 node_trace 的记录。
    """
    return {"node_trace": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# MemorySaver：内存中的检查点保存器，将图编译时传入
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# thread_id 用于标识一次独立的执行线程（类似会话 ID）
# 同一个 thread_id 下的状态会被连续追踪
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"input": "foo"}, config)
# get_state_history：获取该线程下所有检查点的状态快照列表
# 可以看到从初始状态到最终状态的完整变化过程
print(list(graph.get_state_history(config)))
