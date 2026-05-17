"""Tests for the Form Agent module.

Note: These tests require OPENAI_API_KEY to be set for integration tests.
Unit tests mock the LLM calls.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.agent import FormAgent


DATA_DIR = Path(__file__).parent.parent / "data"


class TestFormAgentUnit:
    """Unit tests for FormAgent (no API calls)."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_init(self):
        """Test agent initialization."""
        agent = FormAgent()
        assert agent.loaded_forms == []

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_load_form(self):
        """Test loading a form file."""
        agent = FormAgent()
        file_path = str(DATA_DIR / "sample_employment_form.txt")
        result = agent.load_form(file_path)

        assert result["metadata"]["filename"] == "sample_employment_form.txt"
        assert result["num_chunks"] > 0
        assert len(result["fields"]) > 0
        assert "sample_employment_form.txt" in agent.loaded_forms

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_load_multiple_forms(self):
        """Test loading multiple forms."""
        agent = FormAgent()

        agent.load_form(str(DATA_DIR / "sample_employment_form.txt"))
        agent.load_form(str(DATA_DIR / "sample_tax_form.txt"))
        agent.load_form(str(DATA_DIR / "sample_medical_form.txt"))

        assert len(agent.loaded_forms) == 3

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    def test_reset(self):
        """Test resetting the agent."""
        agent = FormAgent()
        agent.load_form(str(DATA_DIR / "sample_employment_form.txt"))
        assert len(agent.loaded_forms) == 1

        agent.reset()
        assert len(agent.loaded_forms) == 0


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set - skipping integration tests",
)
class TestFormAgentIntegration:
    """Integration tests that make actual API calls."""

    def setup_method(self):
        self.agent = FormAgent()
        self.agent.load_form(str(DATA_DIR / "sample_employment_form.txt"))

    def teardown_method(self):
        self.agent.reset()

    def test_ask_single_form(self):
        """Test asking a question about a single form."""
        answer = self.agent.ask("What position is the applicant applying for?")
        assert "Senior Software Engineer" in answer

    def test_ask_specific_form(self):
        """Test asking about a specific form by name."""
        answer = self.agent.ask(
            "What is the applicant's name?",
            form_name="sample_employment_form.txt",
        )
        assert "Sarah Johnson" in answer

    def test_summarize_single_form(self):
        """Test summarizing a single form."""
        summary = self.agent.summarize(form_name="sample_employment_form.txt")
        assert len(summary) > 100
        assert "Sarah" in summary or "Johnson" in summary

    def test_holistic_analysis(self):
        """Test holistic analysis across multiple forms."""
        self.agent.load_form(str(DATA_DIR / "sample_tax_form.txt"))
        self.agent.load_form(str(DATA_DIR / "sample_medical_form.txt"))

        analysis = self.agent.holistic_analysis(
            "What personal information is common across all forms?"
        )
        assert len(analysis) > 100
