"""
查询路由（Query Routing）示例。

根据用户问题的意图，将其路由到最相关的数据源（举办城市、奖牌榜或通用描述）。
使用结构化输出（Structured Output）让 LLM 返回预定义的路由标签。

使用方法：
    python 04_query_routing.py \\
        --question "里约奥运会哪个国家获得的金牌最多?"
"""
import argparse
from typing import Literal
from pprint import pprint

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START

dotenv.load_dotenv()


class State(TypedDict):
    """LangGraph 状态定义，包含原始问题、检索上下文、答案和路由目标数据源。"""
    question: str
    context: str
    answer: str
    datasource: str


class RouteQuery(BaseModel):
    """路由结果模型，将用户问题映射到最相关的数据源。"""

    datasource: Literal["hosts", "medals", "general_description"] = Field(
        ...,
        description="Given a user query, route it to the most relevant datasource for answering their question",
    )


system = """You are an expert at routing a user question to the appropriate datasource.
Based on the information needed to answer the user's question, you route the question to the most relevant datasource. There are three datasources.
- `medals` which contains all the detailed information related to medals. 
- `hosts` which contains all the detailed information related to the host countries.
- `general` which contains general wiki pages about the Olympics from 1980 to 2024. 
"""

route_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


def route(state: State):
    """根据用户问题选择最合适的数据源。

    调用带结构化输出的LLM，将问题路由到以下三个数据源之一：
    - medals：奖牌榜相关问题
    - hosts：举办城市相关问题
    - general：其他关于奥林匹克运动会的问题
    """
    prompt = route_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response = llm.invoke(prompt)
    return {
        "datasource": response.datasource,
    }


def main():
    parser = argparse.ArgumentParser(
        description="查询路由：根据问题意图将用户查询路由到最相关的数据源。"
    )
    parser.add_argument(
        "--question", required=True, type=str,
        help="用户输入的查询问题。"
    )
    args = parser.parse_args()

    global llm

    llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.1-Terminus", temperature=0).with_structured_output(RouteQuery)

    graph_builder = StateGraph(State)
    graph_builder.add_node(route)
    graph_builder.add_edge(START, "route")
    graph = graph_builder.compile()

    result = graph.invoke({"question": args.question})
    pprint(result)

    # 示例输出：
    # {'datasource': 'medals', 'question': '里约奥运会哪个国家获得的金牌最多?'}


if __name__ == "__main__":
    main()
