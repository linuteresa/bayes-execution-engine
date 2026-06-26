from typing import List, Union

from pydantic import BaseModel, Field


class Plan(BaseModel):
    steps: List[str] = Field(..., description="Different steps to follow, should be in sorted order")

class Response(BaseModel):
    response: str = Field(..., description="Final response to the user query")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(..., description="Either a final response or a new plan")