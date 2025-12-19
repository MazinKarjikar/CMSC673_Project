#!/usr/bin/env python3
"""
Unit tests for training_tooling module.

Tests answer extraction and reward computation utilities.

Run with: python -m pytest tests/test_training.py -v
"""

import re
import pytest


def extract_answer(text: str):
    """Extract answer from GSM8K format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    return numbers[-1].replace(',', '') if numbers else None


def compute_reward(is_correct: bool, total_tokens: int, max_tokens: int = 1000) -> float:
    """Compute simple reward."""
    correctness = 1.0 if is_correct else 0.0
    efficiency = 1.0 - min(total_tokens / max_tokens, 1.0)
    return correctness + 0.2 * efficiency


class TestAnswerExtraction:
    """Tests for answer extraction."""

    def test_gsm8k_format(self):
        assert extract_answer("Solution\n#### 42") == "42"

    def test_fallback_to_last(self):
        assert extract_answer("First 10, then 20") == "20"

    def test_with_comma(self):
        assert extract_answer("#### 1,234") == "1234"


class TestRewardComputation:
    """Tests for reward computation."""

    def test_correct_answer(self):
        reward = compute_reward(is_correct=True, total_tokens=0)
        assert reward == 1.2  # 1.0 + 0.2*1.0

    def test_incorrect_answer(self):
        reward = compute_reward(is_correct=False, total_tokens=500)
        assert reward == 0.1  # 0.0 + 0.2*0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
