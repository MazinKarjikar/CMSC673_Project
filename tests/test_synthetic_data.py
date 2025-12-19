#!/usr/bin/env python3
"""
Unit tests for synthetic_data_tooling module.

Tests JSON parsing and safety filter utilities.

Run with: python -m pytest tests/test_synthetic_data.py -v
"""

import json
import re
import pytest


def parse_json_response(text: str):
    """Parse JSON from LLM output."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def is_safe_plan(plan: dict) -> bool:
    """Check if plan is safe for AI execution (with some dummy example words)"""
    if not plan.get("decompose", False):
        return True
    subproblems = plan.get("subproblems", [])
    dump = json.dumps(subproblems).lower()
    unsafe_words = ["patient", "surgery", "microscope", "weigh"]
    return not any(word in dump for word in unsafe_words)


class TestJsonParsing:
    """Tests for JSON parsing."""

    def test_plan_json(self):
        result = parse_json_response('{"decompose": true, "subproblems": ["A"]}')
        assert result["decompose"] is True

    def test_json_in_markdown(self):
        result = parse_json_response('```json\n{"decompose": false}\n```')
        assert result["decompose"] is False


class TestSafetyFilter:
    """Tests for safety filter."""

    def test_atomic_is_safe(self):
        assert is_safe_plan({"decompose": False}) is True

    def test_math_is_safe(self):
        plan = {"decompose": True, "subproblems": ["Calculate total"]}
        assert is_safe_plan(plan) is True

    def test_medical_is_unsafe(self):
        plan = {"decompose": True, "subproblems": ["Examine patient"]}
        assert is_safe_plan(plan) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
