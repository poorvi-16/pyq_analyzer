import streamlit as st
import os
import tempfile
from rag import ask_question
from ingest import split_documents, create_vector_store
from langchain_community.document_loaders import PyPDFLoader

# Page config
st.set_page_config(
    page_title="PYQ Analyzer",
    page_icon="📚",
    layout="wide"
)

# Auto-detect subjects from data folder
DATA_PATH = "data"
def get_available_subjects():
    if not os.path.exists(DATA_PATH):
        return ["All"]
    return ["All"] + [
        folder for folder in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, folder))
    ]

# Header
st.title("📚 Previous Year Question Paper Analyzer")
st.markdown("*Analyze VTU engineering question papers — find patterns, important topics, and practice questions*")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    available_subjects = get_available_subjects()
    subject = st.selectbox(
        "Select Subject",
        options=available_subjects,
        index=0
    )

    st.divider()

    # ---- PDF UPLOAD SECTION ----
    st.markdown("### 📤 Upload New Question Papers")
    
    new_subject = st.text_input(
        "Subject Name",
        placeholder="e.g. DSA, SE, FLAT"
    ).strip().upper()

    doc_type = st.radio(
        "Document Type",
        options=["qp", "notes"],
        format_func=lambda x: "📝 Question Paper" if x == "qp" else "📖 Notes/Textbook"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("➕ Add to System", use_container_width=True):
        if not new_subject:
            st.error("Please enter a subject name!")
        elif not uploaded_files:
            st.error("Please upload at least one PDF!")
        else:
            with st.spinner(f"Adding {new_subject} to the system..."):
                try:
                    # Create subject folder
                    subject_path = os.path.join(DATA_PATH, new_subject)
                    os.makedirs(subject_path, exist_ok=True)

                    all_docs = []

                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_path = os.path.join(subject_path, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load and tag the PDF
                        try:
                            loader = PyPDFLoader(temp_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["subject"] = new_subject
                                doc.metadata["source_file"] = uploaded_file.name
                                doc.metadata["doc_type"] = doc_type
                            all_docs.extend(docs)
                            st.write(f"✅ Loaded: {uploaded_file.name} ({len(docs)} pages)")
                        except Exception as e:
                            st.warning(f"⚠️ Skipped {uploaded_file.name}: {e}")

                    if all_docs:
                        # Split and add to vector store
                        chunks = split_documents(all_docs)
                        create_vector_store(chunks)
                        st.success(f"🎉 {new_subject} added successfully! ({len(all_docs)} pages, {len(chunks)} chunks)")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What are the most frequently asked topics?",
        "Which units are most important for exams?",
        "Generate 5 practice questions",
        "What questions repeat every year?",
        "Summarize important topics in unit 1",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

    st.divider()

    # Loaded subjects
    st.markdown("### 📂 Loaded Subjects")
    for subj in available_subjects[1:]:
        st.success(f"✅ {subj}")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Ask a question about your question papers:",
        value=st.session_state.get("question", ""),
        height=100,
        placeholder="e.g. What are the most repeated topics in OS?"
    )
    ask_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🎯 Quick Tips")
    st.info("Upload your own PDFs using the sidebar")
    st.info("Select a specific subject for accurate results")
    st.info("Ask about specific units for targeted prep")
    st.info("Request practice questions to test yourself")

# Answer section
if ask_button and question.strip():
    with st.spinner(f"Analyzing {subject} question papers..."):
        try:
            answer, sources = ask_question(question, subject_filter=subject)

            st.divider()
            st.markdown("## 📝 Analysis")
            st.markdown(answer)

            if sources:
                st.divider()
                st.markdown("### 📄 Sources Referenced")
                for source in sources:
                    st.markdown(f"- {source}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try again in a few seconds.")

elif ask_button and not question.strip():
    st.warning("Please enter a question first!")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray'>Built with LangChain + Groq + ChromaDB | VTU PYQ Analyzer</div>",
    unsafe_allow_html=True
)