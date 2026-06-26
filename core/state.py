import operator
from typing import Annotated, List, Tuple, TypedDict

class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]],operator.add]
    response: str
    confidence_score: float