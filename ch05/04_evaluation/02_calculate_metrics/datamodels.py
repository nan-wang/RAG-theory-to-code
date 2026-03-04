"""定义指标计算阶段使用的关键要点数据模型。"""

from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    """表示一个带有验证标签的答案关键要点。"""

    question: str = Field(..., description="The question.")
    answer: str = Field(..., description="The answer.")
    keypoint: str = Field(
        ...,
        description="The keypoint related to the question which should be covered by the answer",
    )
    label: str = Field(
        "Relevant",
        description="The label indicating whether the answer covers the keypoint.",
    )
