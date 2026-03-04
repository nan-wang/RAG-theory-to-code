"""LangGraph 基础 - State（状态）、Node（节点）与 Edge（边）

本脚本演示 LangGraph 中 StateGraph 的三大核心概念：
- State（状态）：使用 TypedDict 定义图的数据结构，在节点间共享和传递
- Node（节点）：执行具体逻辑的函数，接收当前状态并返回更新
- Edge（边）：定义节点之间的执行顺序

注意：默认情况下，节点返回的状态值会覆盖（而非追加）同名字段。
"""

from langgraph.graph import StateGraph, START

from typing_extensions import TypedDict


class State(TypedDict):
    """图的状态定义。

    Attributes:
        step_counter: 记录已执行的节点数量。
        step_list: 记录已执行节点的名称列表。
    """

    step_counter: int
    step_list: list[str]


def node_1(state: State):
    """节点1：将步骤计数器加1，并在步骤列表中记录自身名称。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 更新后的状态字段（覆盖原值）。
    """
    return {
        "step_counter": state.get("step_counter", 0) + 1,
        "step_list": state.get("step_list", []) + [f"node_1"],
    }


def node_2(state: State):
    """节点2：将步骤计数器加1，并在步骤列表中记录自身名称。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 更新后的状态字段（覆盖原值）。
    """
    return {
        "step_counter": state.get("step_counter", 0) + 1,
        "step_list": state.get("step_list", []) + [f"node_2"],
    }


# 构建状态图
graph_builder = StateGraph(state_schema=State)
# 添加节点（函数名自动作为节点名称）
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
# 添加边：START -> node_1 -> node_2，定义执行顺序
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
# 编译图为可执行的应用
app = graph_builder.compile()
output = app.invoke({"step_counter": 0, "step_list": []})
print(output)
