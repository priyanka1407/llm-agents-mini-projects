"""
Multi-agent system: Planner -> Researcher -> Writer <-> Critic

Each "agent" is just the same LLM with a different system prompt (role),
its own tools, and its own small context. A plain Python orchestrator
decides who runs next and passes results between them.

Run:  python orchestrator.py                       (mock mode if no API key)
      python orchestrator.py "Vector databases"
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llm import (LLM, text_of, run_tool_loop, mock_text,  # noqa: E402
                        mock_tool_calls, last_tool_results)

# ---------------- Shared knowledge base (the Researcher's tool) ----------------

KNOWLEDGE = [
    "RAG (retrieval-augmented generation) is a technique where an LLM first retrieves relevant documents and then answers using them.",
    "RAG works step by step: chunk documents, create embeddings, store them in a vector database, retrieve the top matches for a question, and add them to the prompt.",
    "Benefits of RAG: answers use fresh, private data and can cite sources, which reduces hallucination.",
    "Risks of RAG: bad chunking or poor retrieval gives the LLM wrong context, so answers can still be wrong.",
    "Vector databases such as Qdrant, Chroma and Weaviate store embeddings and find nearest neighbours quickly.",
    "Vector databases work by indexing embeddings (for example with HNSW) so similarity search is fast at scale.",
    "Benefits of vector databases: semantic search by meaning, metadata filtering; risks: cost and stale embeddings.",
]
STOP = {"what", "is", "how", "does", "the", "of", "and", "are", "a", "work", "step", "by"}


def search_knowledge(query: str) -> str:
    words = set(re.findall(r"\w+", query.lower())) - STOP
    scored = sorted(KNOWLEDGE, key=lambda f: -len(words & set(re.findall(r"\w+", f.lower()))))
    return " ".join(scored[:2])


RESEARCH_TOOLS = [{
    "name": "search_knowledge",
    "description": "Search the internal knowledge base. Returns the 2 most relevant facts.",
    "input_schema": {"type": "object", "additionalProperties": False,
                     "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "strict": True,
}]

# ---------------- Agents = role prompt (+ optional tools) ----------------


class Agent:
    def __init__(self, llm, name, system, tools=None, tool_funcs=None):
        self.llm, self.name, self.system = llm, name, system
        self.tools, self.tool_funcs = tools, tool_funcs

    def run(self, task: str) -> str:
        print(f"\n--- {self.name} ---")
        if self.tools:  # agent with tools -> same loop as project 01
            out = run_tool_loop(self.llm, self.system, task, self.tools, self.tool_funcs)
        else:           # plain agent -> one LLM call
            out = text_of(self.llm.create(self.system, [{"role": "user", "content": task}]))
        print(out)
        return out


def build_team(llm):
    return {
        "planner": Agent(llm, "PLANNER", "ROLE: planner. Break the topic into exactly 3 short research "
                         "questions. Output one question per line, nothing else."),
        "researcher": Agent(llm, "RESEARCHER", "ROLE: researcher. Use search_knowledge to answer the "
                            "question. Reply with 1-2 sentences of notes, facts only.",
                            RESEARCH_TOOLS, {"search_knowledge": search_knowledge}),
        "writer": Agent(llm, "WRITER", "ROLE: writer. Write a short, clear article (under 150 words) "
                        "from the notes. If critic feedback is given, fix exactly those points."),
        "critic": Agent(llm, "CRITIC", "ROLE: critic. Review the article. If it is accurate, clear and "
                        "ends with a conclusion, reply exactly APPROVED. Otherwise reply "
                        "REVISE: <specific feedback>."),
    }

# ---------------- Orchestrator (plain Python control flow) ----------------


def orchestrate(llm, topic, max_revisions=2):
    team = build_team(llm)

    # 1. Plan
    plan = team["planner"].run(f"Topic: {topic}")
    questions = [q.strip("-• 0123456789.").strip() for q in plan.splitlines() if q.strip()][:3]

    # 2. Research each sub-question (independent -> could run in parallel)
    notes = [team["researcher"].run(q) for q in questions]
    notes_text = "\n".join(f"- {n}" for n in notes)

    # 3. Write, then critic loop (reflection)
    draft = team["writer"].run(f"Topic: {topic}\nNotes:\n{notes_text}")
    for round_no in range(1, max_revisions + 1):
        review = team["critic"].run(f"Article:\n{draft}")
        if review.strip().upper().startswith("APPROVED"):
            print(f"\n[orchestrator] approved after {round_no} review(s)")
            break
        draft = team["writer"].run(f"Topic: {topic}\nNotes:\n{notes_text}\n"
                                   f"Previous draft:\n{draft}\nCritic feedback: {review}")
    return draft

# ---------------- Offline mock brain (one fake LLM, many roles) ----------------


def mock_brain(system, messages, tools):
    task = messages[0]["content"]
    if "ROLE: planner" in system:
        topic = task.replace("Topic:", "").strip()
        return mock_text(f"What is {topic}?\nHow does {topic} work step by step?\n"
                         f"What are the benefits and risks of {topic}?")
    if "ROLE: researcher" in system:
        results = last_tool_results(messages)
        if results:
            return mock_text(results["search_knowledge"])
        return mock_tool_calls([("search_knowledge", {"query": task})])
    if "ROLE: writer" in system:
        topic = task.splitlines()[0].replace("Topic:", "").strip()
        sentences = re.split(r"(?<=\.)\s+", " ".join(l[2:] for l in task.splitlines() if l.startswith("- ")))
        notes = " ".join(dict.fromkeys(sentences))  # drop repeated facts, keep order
        article = f"{topic}: an overview. {notes}"
        if "Critic feedback" in task:
            article += f" Conclusion: {topic} is powerful when the data and retrieval are good."
        return mock_text(article)
    if "ROLE: critic" in system:
        if "Conclusion" in task:
            return mock_text("APPROVED")
        return mock_text("REVISE: add a one-sentence conclusion at the end.")
    return mock_text("(unknown role)")


if __name__ == "__main__":
    llm = LLM(mock_brain)
    topic = " ".join(sys.argv[1:]) or "RAG"
    final = orchestrate(llm, topic)
    print(f"\n===== FINAL ARTICLE =====\n{final}")
