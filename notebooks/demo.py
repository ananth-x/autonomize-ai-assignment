"""
Demonstration Script - Intelligent Form Agent
==============================================

This script demonstrates the three core capabilities:
1. Answering a question from a single form
2. Generating a summary of one form
3. Providing holistic answers across multiple forms

Run: python -m notebooks.demo
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import FormAgent


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("🧠 Intelligent Form Agent - Demonstration")
    print("=" * 60)

    # Initialize agent
    agent = FormAgent()

    # Load all sample forms
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    forms = [
        "sample_employment_form.txt",
        "sample_tax_form.txt",
        "sample_medical_form.txt",
    ]

    print("\n📂 Loading forms...")
    for form in forms:
        path = os.path.join(data_dir, form)
        result = agent.load_form(path)
        print(f"  ✅ {form} - {result['num_chunks']} chunks, {len(result['fields'])} fields")

    # =========================================================
    # DEMO 1: Single Form Question Answering
    # =========================================================
    separator("DEMO 1: Single Form Question Answering")

    questions = [
        ("sample_employment_form.txt", "What is the applicant's desired salary and current salary?"),
        ("sample_tax_form.txt", "What is the total refund amount and how was it calculated?"),
        ("sample_medical_form.txt", "What medications is the patient currently taking and for what conditions?"),
    ]

    for form_name, question in questions:
        print(f"📄 Form: {form_name}")
        print(f"❓ Question: {question}")
        answer = agent.ask(question, form_name=form_name)
        print(f"💡 Answer: {answer}")
        print()

    # =========================================================
    # DEMO 2: Form Summarization
    # =========================================================
    separator("DEMO 2: Form Summarization")

    print("📄 Summarizing: sample_employment_form.txt\n")
    summary = agent.summarize(form_name="sample_employment_form.txt")
    print(f"📝 Summary:\n{summary}")

    # =========================================================
    # DEMO 3: Holistic Analysis Across Multiple Forms
    # =========================================================
    separator("DEMO 3: Holistic Analysis Across Multiple Forms")

    holistic_questions = [
        "What personal information (names, addresses, dates) is collected across all three forms? Compare the types of data each form requires.",
        "What are the key differences in the purpose and structure of these three forms?",
    ]

    for question in holistic_questions:
        print(f"❓ Question: {question}\n")
        analysis = agent.holistic_analysis(question)
        print(f"🔬 Analysis:\n{analysis}")
        print()

    # =========================================================
    # BONUS: Field Extraction & Explanation
    # =========================================================
    separator("BONUS: Field Extraction & Explanation")

    print("📄 Explaining fields from: sample_medical_form.txt\n")
    explanation = agent.explain_fields("sample_medical_form.txt")
    print(f"🏷️  Explanation:\n{explanation}")

    # Cleanup
    agent.reset()
    print(f"\n{'='*60}")
    print("  ✅ Demonstration Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
