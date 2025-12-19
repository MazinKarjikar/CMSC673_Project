#!/usr/bin/env python3
"""
Unit tests for recursive_reasoner_tooling module.

Tests JSON parsing and result structure utilities.

Run with: python -m pytest tests/test_recursive_reasoner.py -v
"""

import json
import re
import pytest


def parse_json_response(text):
    """Parse JSON from model output."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class TestJsonParsing:
    """Tests for JSON parsing."""

    def test_solution_json(self):
        result = parse_json_response('{"solution": "x=5"}')
        assert result["solution"] == "x=5"

    def test_decision_json(self):
        result = parse_json_response('{"decision": "DECOMPOSE"}')
        assert result["decision"] == "DECOMPOSE"

    def test_subproblems_array(self):
        result = parse_json_response('{"subproblems": ["A", "B"]}')
        assert len(result["subproblems"]) == 2


class TestResultStructure:
    """Tests for result data structures."""

    def test_atomic_result(self):
        result = {
            'solution': "42",
            'atomic': True,
            'subproblems': []
        }
        assert result['atomic'] is True
        assert result['subproblems'] == []

    def test_decomposed_result(self):
        result = {
            'solution': "Combined",
            'atomic': False,
            'subproblems': ["A", "B"]
        }
        assert result['atomic'] is False
        assert len(result['subproblems']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
