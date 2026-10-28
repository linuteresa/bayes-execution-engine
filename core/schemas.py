"""Strict contracts for the JSON emitted by planner and replanner models."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints

Step = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1,
                                       max_length=4000)]
Answer = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1,
                                         max_length=32000)]


class Plan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    steps: list[Step] = Field(min_length=1, max_length=64)


class Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    response: Answer


class PlanAction(Plan):
    action: Literal["plan"]
    # An empty remaining plan is a valid instruction to finish with gathered results.
    steps: list[Step] = Field(max_length=64)


class ResponseAction(Response):
    action: Literal["response"]


Decision = Annotated[Union[PlanAction, ResponseAction], Field(discriminator="action")]


class Act(RootModel[Decision]):
    pass
