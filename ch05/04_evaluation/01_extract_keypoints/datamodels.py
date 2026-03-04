"""定义关键信息要点提取的 Pydantic 数据模型。"""

from pydantic import BaseModel, Field


class KeyPoints(BaseModel):
    """从上下文中提取的关键信息要点列表。"""

    keypoints: list = Field(
        ..., description="The keypoints extracted from the context."
    )
