"""LangGraph 消息管理 - 使用 add_messages 归约器

本脚本演示 LangGraph 中专门用于消息管理的 add_messages 归约器。
与普通的 operator.add 不同，add_messages 具备消息去重和更新功能：
- 如果新消息的 ID 与已有消息相同，则更新该消息
- 如果是新 ID，则追加到列表末尾

这是构建聊天机器人时管理对话历史的推荐方式。
"""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """图的状态定义。

    Attributes:
        msg_list: 消息列表，使用 add_messages 归约器管理。
            AnyMessage 是所有消息类型的联合类型，
            包括 HumanMessage、AIMessage、SystemMessage 等。
    """

    msg_list: Annotated[list[AnyMessage], add_messages]


def node_1(state: State):
    """节点1：添加一条用户消息。

    使用字典格式创建消息，type 指定消息类型（human/assistant/system）。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 包含新消息的字典。
    """
    return {"msg_list": [{"type": "human", "content": "你好！"}]}


def node_2(state: State):
    """节点2：添加一条 AI 助手回复消息。

    Args:
        state: 当前图的状态。

    Returns:
        dict: 包含新消息的字典。
    """
    return {"msg_list": [{"type": "assistant", "content": "你好，有什么可以帮助您的?"}]}


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
app = graph_builder.compile()
output = app.invoke({"msg_list": []})
print(output)
