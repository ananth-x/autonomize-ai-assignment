"""Command-line interface for the Intelligent Form Agent.

Provides an interactive REPL for loading forms, asking questions,
and generating summaries.
"""

import sys
import os
from pathlib import Path

from .agent import FormAgent


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  🧠 Intelligent Form Agent")
    print("  Read, Extract, and Explain")
    print("=" * 60)
    print()


def print_help():
    """Print available commands."""
    print("""
Available Commands:
  load <file_path>       Load a form file (PDF, image, DOCX, TXT)
  ask <question>         Ask a question about loaded forms
  ask:<form> <question>  Ask about a specific form
  summary                Summarize all loaded forms
  summary:<form>         Summarize a specific form
  holistic <question>    Cross-form analysis question
  fields <form>          Explain extracted fields from a form
  forms                  List loaded forms
  reset                  Clear all loaded forms
  help                   Show this help message
  quit / exit            Exit the agent

Examples:
  load ./data/tax_form.pdf
  ask What is the total income reported?
  ask:tax_form.pdf What deductions are listed?
  summary:tax_form.pdf
  holistic What common fields appear across all forms?
""")


def run_cli():
    """Run the interactive CLI."""
    print_banner()

    try:
        agent = FormAgent()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    print("Type 'help' for available commands, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("🤖 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Parse command
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        try:
            if command in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            elif command == "help":
                print_help()

            elif command == "load":
                if not args:
                    print("Usage: load <file_path>")
                    continue
                file_path = args.strip()
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    continue
                print(f"📄 Loading '{file_path}'...")
                result = agent.load_form(file_path)
                print(f"✅ Loaded: {result['metadata']['filename']}")
                print(f"   Pages: {result['metadata'].get('pages', 'N/A')}")
                print(f"   Chunks indexed: {result['num_chunks']}")
                print(f"   Fields extracted: {len(result['fields'])}")
                if result['fields']:
                    print("   Sample fields:")
                    for k, v in list(result['fields'].items())[:5]:
                        print(f"     - {k}: {v}")

            elif command == "ask" or command.startswith("ask:"):
                if not args:
                    print("Usage: ask <question> or ask:<form_name> <question>")
                    continue
                form_name = None
                if ":" in command:
                    form_name = command.split(":", 1)[1]
                print("🔍 Thinking...")
                answer = agent.ask(args, form_name=form_name)
                print(f"\n📋 Answer:\n{answer}\n")

            elif command == "summary" or command.startswith("summary:"):
                form_name = None
                if ":" in command:
                    form_name = command.split(":", 1)[1]
                elif args:
                    form_name = args
                print("📝 Generating summary...")
                summary = agent.summarize(form_name=form_name)
                print(f"\n📋 Summary:\n{summary}\n")

            elif command == "holistic":
                if not args:
                    print("Usage: holistic <question>")
                    continue
                print("🔬 Analyzing across forms...")
                analysis = agent.holistic_analysis(args)
                print(f"\n📋 Holistic Analysis:\n{analysis}\n")

            elif command == "fields":
                if not args:
                    print("Usage: fields <form_name>")
                    continue
                print("🏷️  Explaining fields...")
                explanation = agent.explain_fields(args)
                print(f"\n📋 Field Explanation:\n{explanation}\n")

            elif command == "forms":
                forms = agent.loaded_forms
                if forms:
                    print(f"📂 Loaded forms ({len(forms)}):")
                    for f in forms:
                        print(f"   - {f}")
                else:
                    print("No forms loaded yet. Use 'load <file_path>' to add forms.")

            elif command == "reset":
                agent.reset()
                print("🗑️  All forms cleared.")

            else:
                # Treat as a question if forms are loaded
                if agent.loaded_forms:
                    print("🔍 Thinking...")
                    answer = agent.ask(user_input)
                    print(f"\n📋 Answer:\n{answer}\n")
                else:
                    print(f"Unknown command: '{command}'. Type 'help' for available commands.")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_cli()
