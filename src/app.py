"""Streamlit UI for the Intelligent Form Agent.

Provides a web-based interface for:
- Uploading form documents
- Asking questions
- Generating summaries
- Cross-form analysis
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from .agent import FormAgent


def init_agent():
    """Initialize or retrieve the agent from session state."""
    if "agent" not in st.session_state:
        st.session_state.agent = FormAgent()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.agent


def main():
    st.set_page_config(
        page_title="Intelligent Form Agent",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Intelligent Form Agent")
    st.caption("Read, Extract, and Explain — Upload forms and get instant insights")

    try:
        agent = init_agent()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.info("Please create a `.env` file with your `OPENAI_API_KEY`. See `.env.example`.")
        st.stop()

    # Sidebar - File Upload & Management
    with st.sidebar:
        st.header("📂 Form Management")

        uploaded_files = st.file_uploader(
            "Upload Forms",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "docx", "txt", "csv"],
            accept_multiple_files=True,
            help="Upload one or more form documents",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in agent.loaded_forms:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=Path(uploaded_file.name).suffix,
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            result = agent.load_form(tmp_path)
                            st.success(f"✅ {uploaded_file.name} loaded ({result['num_chunks']} chunks)")
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                        finally:
                            os.unlink(tmp_path)

        # Show loaded forms
        st.divider()
        st.subheader("Loaded Forms")
        if agent.loaded_forms:
            for form_name in agent.loaded_forms:
                st.text(f"📄 {form_name}")
        else:
            st.info("No forms loaded yet. Upload files above.")

        # Reset button
        st.divider()
        if st.button("🗑️ Clear All Forms", type="secondary"):
            agent.reset()
            st.session_state.chat_history = []
            st.rerun()

    # Main content area - Tabs
    tab_qa, tab_summary, tab_holistic, tab_fields = st.tabs([
        "💬 Ask Questions",
        "📝 Summarize",
        "🔬 Holistic Analysis",
        "🏷️ Field Extraction",
    ])

    # Tab 1: Question Answering
    with tab_qa:
        st.subheader("Ask Questions About Your Forms")

        if not agent.loaded_forms:
            st.info("Upload forms in the sidebar to get started.")
        else:
            # Form selector
            form_options = ["All Forms"] + agent.loaded_forms
            selected_form = st.selectbox(
                "Target Form",
                form_options,
                key="qa_form_select",
            )

            question = st.text_input(
                "Your Question",
                placeholder="e.g., What is the applicant's name?",
                key="qa_input",
            )

            if st.button("Ask", type="primary", key="qa_button"):
                if question:
                    form_name = None if selected_form == "All Forms" else selected_form
                    with st.spinner("Thinking..."):
                        answer = agent.ask(question, form_name=form_name)
                    st.markdown("**Answer:**")
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "type": "qa",
                        "question": question,
                        "answer": answer,
                        "form": selected_form,
                    })

    # Tab 2: Summarization
    with tab_summary:
        st.subheader("Form Summaries")

        if not agent.loaded_forms:
            st.info("Upload forms in the sidebar to get started.")
        else:
            form_options = ["All Forms"] + agent.loaded_forms
            selected_form = st.selectbox(
                "Form to Summarize",
                form_options,
                key="summary_form_select",
            )

            if st.button("Generate Summary", type="primary", key="summary_button"):
                form_name = None if selected_form == "All Forms" else selected_form
                with st.spinner("Generating summary..."):
                    summary = agent.summarize(form_name=form_name)
                st.markdown("**Summary:**")
                st.markdown(summary)

    # Tab 3: Holistic Analysis
    with tab_holistic:
        st.subheader("Cross-Form Analysis")
        st.caption("Ask questions that span multiple forms to find patterns and insights.")

        if len(agent.loaded_forms) < 2:
            st.info("Load at least 2 forms to use holistic analysis.")
        else:
            question = st.text_input(
                "Analysis Question",
                placeholder="e.g., What common information appears across all forms?",
                key="holistic_input",
            )

            if st.button("Analyze", type="primary", key="holistic_button"):
                if question:
                    with st.spinner("Analyzing across forms..."):
                        analysis = agent.holistic_analysis(question)
                    st.markdown("**Analysis:**")
                    st.markdown(analysis)

    # Tab 4: Field Extraction
    with tab_fields:
        st.subheader("Extracted Fields")

        if not agent.loaded_forms:
            st.info("Upload forms in the sidebar to get started.")
        else:
            selected_form = st.selectbox(
                "Select Form",
                agent.loaded_forms,
                key="fields_form_select",
            )

            if selected_form:
                form_data = agent._loaded_forms.get(selected_form, {})
                fields = form_data.get("fields", {})

                if fields:
                    st.markdown("**Extracted Key-Value Pairs:**")
                    for key, value in fields.items():
                        st.text(f"  {key}: {value}")

                    st.divider()
                    if st.button("Explain Fields", type="primary", key="fields_button"):
                        with st.spinner("Explaining fields..."):
                            explanation = agent.explain_fields(selected_form)
                        st.markdown("**Explanation:**")
                        st.markdown(explanation)
                else:
                    st.warning("No structured fields were extracted from this form.")
                    st.caption(
                        "This may happen with unstructured documents. "
                        "Try the Q&A tab to ask specific questions instead."
                    )

    # Chat History (collapsible)
    if st.session_state.chat_history:
        with st.expander("📜 Session History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {entry['question']}")
                st.markdown(f"*A:* {entry['answer'][:200]}...")
                st.divider()


if __name__ == "__main__":
    main()
