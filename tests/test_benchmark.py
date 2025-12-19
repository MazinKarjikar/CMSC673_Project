#!/usr/bin/env python3
"""
Unit tests for benchmark_tooling module.

Tests JSON parsing and number extraction utilities.

Run with: python -m pytest tests/test_benchmark.py -v
"""

import json
import re
import pytest


def parse_json_response(output: str):
    """Parse JSON from model output."""
    if not output:
        return None
    output = re.sub(r"```(?:json)?", "", output)
    output = re.sub(r"```", "", output)
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract_number(text: str):
    """Extract numerical answer from text."""
    if text is None:
        return None
    try:
        return float(str(text).replace(',', '').replace('$', '').strip())
    except:
        pass
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', str(text))
    return float(numbers[-1].replace(',', '')) if numbers else None


class TestParseJsonResponse:
    """Tests for JSON parsing."""

    def test_simple_json(self):
        assert parse_json_response('{"answer": 42}') == {"answer": 42}

    def test_json_in_markdown(self):
        result = parse_json_response('```json\n{"value": 5}\n```')
        assert result["value"] == 5

    def test_empty_returns_none(self):
        assert parse_json_response("") is None


class TestExtractNumber:
    """Tests for number extraction."""

    def test_simple_integer(self):
        assert extract_number("42") == 42.0

    def test_currency(self):
        assert extract_number("$45.50") == 45.50

    def test_number_in_text(self):
        assert extract_number("The answer is 100") == 100.0

    def test_none_input(self):
        assert extract_number(None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
