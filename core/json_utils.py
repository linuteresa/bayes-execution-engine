"""
Tolerant JSON extraction for small-LLM output.

Local instruction-tuned models frequently wrap JSON in ```` ```json ```` fences, prefix
it with a sentence ("Sure, here is the plan:"), or trail it with commentary. The old
`text.find('{') ... text.rfind('}')` approach grabs everything between the first and last
brace, which breaks the moment there is more than one object or any stray brace.

`extract_json` instead finds the first **balanced** JSON object using a string-aware
brace scan, after stripping code fences. This is what stops the planner/replanner loop
from silently producing an empty plan when the model is slightly chatty.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Return the first balanced JSON object found in ``text``, or ``None``.

    Tries fenced code blocks first (most models put their answer there), then the raw
    text. Brace matching is string-aware so braces inside string literals don't confuse
    the scan.
    """
    if not text:
        return None

    candidates = [m.strip() for m in _FENCE.findall(text)]
    candidates.append(text)

    for candidate in candidates:
        obj = _first_balanced_object(candidate)
        if obj is not None:
            return obj
    return None


def _first_balanced_object(s: str) -> Optional[dict[str, Any]]:
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
        start = s.find("{", start + 1)
    return None
