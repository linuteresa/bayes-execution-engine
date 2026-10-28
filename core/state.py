import operator
from typing import Annotated, Dict, List, Tuple, TypedDict


class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]],operator.add]
    step_confidences: Annotated[List[float], operator.add]
    step_evidence: Annotated[List[Dict[str, int]], operator.add]
    response: str
    confidence_score: float