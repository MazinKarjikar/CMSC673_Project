#!/usr/bin/env python3
"""
Unit tests for serving_tooling module.

Tests JSON parsing and utility functions.

Run with: python -m pytest tests/test_serving.py -v
"""

import json
import re
import pytest


def parse_json(text: str):
    """Parse JSON from model output."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


class TestJsonParsing:
    """Tests for JSON parsing."""

    def test_simple_json(self):
        assert parse_json('{"result": 5}') == {"result": 5}

    def test_json_in_text(self):
        result = parse_json('Answer: {"value": 10}')
        assert result["value"] == 10


class TestAnsiStripping:
    """Tests for ANSI code removal."""

    def test_strip_colors(self):
        text = "\033[91mRed\033[0m"
        assert strip_ansi(text) == "Red"

    def test_plain_text_unchanged(self):
        assert strip_ansi("Hello") == "Hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
