from typing import List, TypedDict, Annotated, Tuple
import operator

class PlanExecuteState(TypedDict):
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]],operator.add]
    response:str