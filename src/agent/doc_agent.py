from typing import List

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


# ==================================================
# Agent setup
# ==================================================

SYSTEM_PROMPT = """
You are a documentation expert AI assistant.

Answer user questions strictly using the provided documentation context.
If the answer is not present in the context, say:
"I could not find the answer in the documentation."
"""


def get_llm(model: str = "llama3.2") -> ChatOllama:
    """Initialize the LLM."""
    return ChatOllama(
        model=model,
        temperature=0.2,
    )


# ==================================================
# Context formatting
# ==================================================

def format_context(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a readable context block.
    """
    context_blocks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("url", "unknown")
        content = doc.page_content.strip()

        context_blocks.append(
            f"[Source {i}]\nURL: {source}\n{content}"
        )

    return "\n\n".join(context_blocks)


# ==================================================
# Agent query execution
# ==================================================

def answer_question(
    query: str,
    retrieved_docs: List[Document],
    llm: ChatOllama,
) -> str:
    """
    Generate an answer using retrieved documentation context.
    """
    context = format_context(retrieved_docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Documentation context:
{context}

Question:
{query}
"""
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()

