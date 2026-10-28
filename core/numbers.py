"""Exact numeric literals used to detect incompatible sampled answers."""

from __future__ import annotations

import re
from fractions import Fraction

_LITERAL = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER = re.compile(rf"(?<!\w)([+-]?{_LITERAL})(?:\s*/\s*([+-]?{_LITERAL}))?(\s*%)?(?!\w)")
_UNIT = re.compile(r"\s*([a-zA-Z]+)\b")
_UNITS = {
    "mm": (Fraction(1, 1000), "m"), "cm": (Fraction(1, 100), "m"),
    "m": (Fraction(1), "m"), "km": (Fraction(1000), "m"),
    "mg": (Fraction(1, 1000), "g"), "g": (Fraction(1), "g"),
    "kg": (Fraction(1000), "g"),
    "ms": (Fraction(1, 1000), "s"), "s": (Fraction(1), "s"),
    "seconds": (Fraction(1), "s"), "minutes": (Fraction(60), "s"),
    "hours": (Fraction(3600), "s"),
    "usd": (Fraction(1), "USD"), "dollars": (Fraction(1), "USD"),
    "eur": (Fraction(1), "EUR"), "euros": (Fraction(1), "EUR"),
}


def numeric_claims(text: str) -> tuple[tuple[Fraction, str], ...]:
    """Preserve signs, multiplicity, decimal/fraction/exponent values and known units.

    This is a conservative literal compatibility check, not a mathematical verifier.
    Equivalent supported units and percent/decimal forms share a representation.
    """
    text = text.replace("−", "-")
    claims = []
    for match in _NUMBER.finditer(text):
        numerator, denominator, percent = match.groups()
        tokens = [numerator] + ([denominator] if denominator else [])
        # Prevent model output such as 1e999999999 from allocating huge integers.
        if any(len(token) > 128 or ("e" in token.lower() and
               abs(int(token.lower().split("e")[1])) > 308) for token in tokens):
            raise ValueError("numeric literal is too large")
        try:
            value = Fraction(numerator.replace(",", ""))
            if denominator:
                value /= Fraction(denominator.replace(",", ""))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid numeric literal") from exc
        unit = ""
        if percent:
            value /= 100
        else:
            following = _UNIT.match(text[match.end():])
            suffix = following.group(1).lower() if following else ""
            if suffix in _UNITS:
                factor, unit = _UNITS[suffix]
                value *= factor
            elif match.start() and text[match.start() - 1] in "$€£":
                unit = {"$": "USD", "€": "EUR", "£": "GBP"}[text[match.start() - 1]]
        claims.append((value, unit))
    return tuple(claims)


def numeric_values(text: str) -> tuple[Fraction, ...]:
    return tuple(value for value, _unit in numeric_claims(text))
