import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

def load_retriever(subject_filter=None):
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    if subject_filter and subject_filter != "All":
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"subject": subject_filter}
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return retriever


def ask_question(question, subject_filter=None):
    retriever = load_retriever(subject_filter)

    prompt_template = PromptTemplate.from_template("""
You are an expert academic assistant helping engineering students prepare for exams.

You have access to TWO types of content:
1. Question Papers (doc_type: qp) — previous year exam questions
2. Study Notes (doc_type: notes) — explanations, concepts, and theory

Using both sources, answer the student's question by:
- Identifying frequently asked exam questions (from question papers)
- Providing detailed explanations and answers (from notes)
- Highlighting which topics from notes are most exam-relevant
- Generating practice questions WITH answers when asked
- Mentioning the source (question paper year or notes file)

Context:
{context}

Student's Question:
{question}

Provide a comprehensive answer with both the questions AND their explanations:
""")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('subject')} - {doc.metadata.get('source_file')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Get sources
    source_docs = retriever.invoke(question)
    seen = set()
    source_list = []
    for doc in source_docs:
        key = f"{doc.metadata.get('subject')} — {doc.metadata.get('source_file')}"
        if key not in seen:
            seen.add(key)
            source_list.append(key)

    answer = chain.invoke(question)
    return answer, source_list


# Quick test
if __name__ == "__main__":
    print("Testing RAG pipeline...\n")
    answer, sources = ask_question(
        "What are the most frequently asked topics in OS?",
        subject_filter="OS"
    )
    print("ANSWER:\n", answer)
    print("\nSOURCES:")
    for s in sources:
        print(" -", s)