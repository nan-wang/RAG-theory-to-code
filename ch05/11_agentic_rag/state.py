"""Agentic RAG LangGraph 图的状态数据类定义。"""

import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


@dataclass(kw_only=True)
class SummaryState:
    """Agentic RAG 图的完整运行状态，贯穿整个搜索-摘要-反思循环。"""

    user_query: str = field(default=None)  # original user query
    search_query: str = field(default=None)  # search query
    web_search_results: Annotated[list, operator.add] = field(
        default_factory=list
    )  # web search results
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    search_loop_count: int = field(default=0)  # search loop count
    running_summary: str = field(default=None)  # final report


@dataclass(kw_only=True)
class SummaryStateInput:
    """图的输入状态，仅包含用户查询。"""

    user_query: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    """图的输出状态，仅包含最终摘要。"""

    running_summary: str = field(default=None)
