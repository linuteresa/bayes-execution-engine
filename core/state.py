import operator
from typing import Annotated, Dict, List, Tuple, TypedDict

#: Marks a plan step that re-runs an earlier one. Lives here, with the plan shape it
#: describes, because the replanner writes it, the graph strips it when keyed by task,
#: and the executor reads it to recover the original step text.
RECHECK_PREFIX = "Re-check: "


class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]],operator.add]
    step_confidences: Annotated[List[float], operator.add]
    step_evidence: Annotated[List[Dict[str, int]], operator.add]
    response: str
    confidence_score: float