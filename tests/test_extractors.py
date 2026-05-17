"""Tests for the document extraction module."""

import os
import pytest
from pathlib import Path

from src.extractors import FormExtractor, StructuredFieldExtractor


# Path to test data
DATA_DIR = Path(__file__).parent.parent / "data"


class TestFormExtractor:
    """Tests for FormExtractor class."""

    def setup_method(self):
        self.extractor = FormExtractor()

    def test_extract_text_file(self):
        """Test extraction from a plain text file."""
        file_path = DATA_DIR / "sample_employment_form.txt"
        result = self.extractor.extract(str(file_path))

        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["extension"] == ".txt"
        assert result["metadata"]["filename"] == "sample_employment_form.txt"
        assert "Sarah Johnson" in result["text"]
        assert "Senior Software Engineer" in result["text"]

    def test_extract_tax_form(self):
        """Test extraction from tax form."""
        file_path = DATA_DIR / "sample_tax_form.txt"
        result = self.extractor.extract(str(file_path))

        assert "John" in result["text"]
        assert "$106,850.00" in result["text"]

    def test_extract_medical_form(self):
        """Test extraction from medical form."""
        file_path = DATA_DIR / "sample_medical_form.txt"
        result = self.extractor.extract(str(file_path))

        assert "Robert Williams" in result["text"]
        assert "lower back pain" in result["text"]

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract("/nonexistent/file.txt")

    def test_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        # Create a temp file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.extractor.extract(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestStructuredFieldExtractor:
    """Tests for StructuredFieldExtractor class."""

    def setup_method(self):
        self.extractor = StructuredFieldExtractor()

    def test_colon_pattern(self):
        """Test extraction of 'Key: Value' patterns."""
        text = "Full Name: John Smith\nEmail: john@example.com\nAge: 30"
        fields = self.extractor.extract_fields(text)

        assert fields["Full Name"] == "John Smith"
        assert fields["Email"] == "john@example.com"
        assert fields["Age"] == "30"

    def test_checkbox_pattern(self):
        """Test extraction of checkbox patterns."""
        text = "[x] I agree to terms\n[ ] I want marketing emails\n[X] I am over 18"
        fields = self.extractor.extract_fields(text)

        assert fields["I agree to terms"] == "Yes"
        assert fields["I want marketing emails"] == "No"
        assert fields["I am over 18"] == "Yes"

    def test_underline_pattern(self):
        """Test extraction of underline-separated patterns."""
        text = "Name _____ John Smith\nDate _____ 01/15/2024"
        fields = self.extractor.extract_fields(text)

        assert fields["Name"] == "John Smith"
        assert fields["Date"] == "01/15/2024"

    def test_empty_text(self):
        """Test extraction from empty text."""
        fields = self.extractor.extract_fields("")
        assert fields == {}

    def test_real_form_extraction(self):
        """Test field extraction on actual sample form."""
        file_path = DATA_DIR / "sample_employment_form.txt"
        with open(file_path) as f:
            text = f.read()

        fields = self.extractor.extract_fields(text)

        # Should extract key fields
        assert "Full Name" in fields
        assert fields["Full Name"] == "Sarah Johnson"
        assert "Position Applied For" in fields
        assert fields["Position Applied For"] == "Senior Software Engineer"
