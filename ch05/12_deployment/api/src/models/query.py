"""API 请求与响应的 Pydantic 数据模型定义。"""

from pydantic import BaseModel


class QueryInput(BaseModel):
    """用户提问的请求体，包含问题文本。"""

    text: str


class QueryOutput(BaseModel):
    """RAG 生成的响应体，包含选中的上下文片段和最终答案。"""

    selected_content: str
    answer: str
