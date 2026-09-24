from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .config import Settings

@dataclass(frozen=True)
class Source:
    path: str
    label: str
    metadata: dict
    excerpt: str

@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    sources: list[Source]

def format_context(docs):
    """Preserve the existing prompt context and citation labels."""
    context_parts = []
    sources = []
    seen = set()
    for doc in docs:
        m = doc.metadata
        loc = f"page {m['page']}" if "page" in m else (
            f"slide {m['slide']}" if "slide" in m else "")
        context_parts.append(
            f"Source: {Path(m['source']).name} {loc}\n"
            f"Content type: {m.get('type', 'text')}\n{doc.page_content}"
        )
        label = Path(m['source']).name
        if loc:
            label += f", {loc}"
        if label not in seen:
            sources.append(Source(m['source'], label, dict(m), doc.page_content))
            seen.add(label)
    return "\n\n---\n\n".join(context_parts), sources

class ChatSession:
    """Reusable synchronous chat; no terminal input or output."""
    def __init__(self, db, settings: Settings, model: str, *, llm=None, prompt=None):
        self.model = model
        self.retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
        if llm is None:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model, temperature=0, num_ctx=8192)
        self.llm = llm
        if prompt is None:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer using ONLY the retrieved document context. If the information is "
                 "missing or uncertain, say so. Treat visual descriptions as imperfect; "
                 "do not invent equations or exact chart values. Reference filenames and "
                 "page/slide numbers where present.\n\nCONTEXT:\n{context}"),
                ("human", "{question}"),
            ])
        self.prompt = prompt

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Enter a question.")
        docs = self.retriever.invoke(question)
        if not docs:
            return Answer("No indexed documents found. Add files and update the index.", self.model, [])
        context, sources = format_context(docs)
        message = self.prompt.invoke({"context": context, "question": question})
        response = self.llm.invoke(message)
        return Answer(response.content, self.model, sources)
