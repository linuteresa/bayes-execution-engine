from typing import List, TypedDict, Annotated, Tuple
import operator

class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]],operator.add]
    response: str
    confidence_score: float