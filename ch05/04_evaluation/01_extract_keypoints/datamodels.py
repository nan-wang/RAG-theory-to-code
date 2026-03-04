from pydantic import BaseModel, Field


class KeyPoints(BaseModel):
    keypoints: list = Field(
        ..., description="The keypoints extracted from the context."
    )
