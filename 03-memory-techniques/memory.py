"""
Four memory techniques for LLM chatbots/agents.

LLMs are stateless: they only "remember" what we put in the prompt.
Memory = deciding WHAT past information goes into the next prompt.

  1. BufferMemory        - keep everything            (simple, grows forever)
  2. WindowMemory        - keep last k turns          (cheap, forgets old facts)
  3. SummaryMemory       - summarise old turns        (compact, some detail lost)
  4. VectorMemory        - long-term store + semantic search, saved to disk
"""
import json
import math
import re
from collections import Counter

from common.llm import text_of


class BufferMemory:
    """Send the full conversation every time."""

    def __init__(self):
        self.messages = []

    def add(self, user, assistant):
        self.messages += [{"role": "user", "content": user},
                          {"role": "assistant", "content": assistant}]

    def context(self, query):
        return "", list(self.messages)        # (extra system text, history messages)


class WindowMemory(BufferMemory):
    """Only the last k turns (1 turn = user + assistant message)."""

    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def context(self, query):
        return "", self.messages[-2 * self.k:]


class SummaryMemory(BufferMemory):
    """Keep recent turns word-for-word; fold older turns into a running summary."""

    def __init__(self, llm, keep_last=2):
        super().__init__()
        self.llm, self.keep_last, self.summary = llm, keep_last, ""

    def add(self, user, assistant):
        super().add(user, assistant)
        if len(self.messages) > 2 * self.keep_last:
            old, self.messages = self.messages[:2], self.messages[2:]
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
            prompt = f"Current summary:\n{self.summary or '(empty)'}\n\nNew lines:\n{transcript}"
            self.summary = text_of(self.llm.create(
                "ROLE: summarizer. Update the summary of this conversation. Keep every fact "
                "about the user (name, location, job, preferences). Be brief.",
                [{"role": "user", "content": prompt}]))

    def context(self, query):
        extra = f"Summary of earlier conversation: {self.summary}" if self.summary else ""
        return extra, list(self.messages)


# ---------- Long-term memory with (toy) embeddings ----------

STOP = {"i", "a", "the", "is", "and", "my", "me", "to", "of", "in", "what", "do", "you", "it", "for", "where", "am"}


def embed(text):
    """Toy embedding: word counts. In production use a real embedding model
    (Voyage, OpenAI, sentence-transformers) and a vector DB (Qdrant, Chroma)."""
    return Counter(w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP)


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class VectorMemory:
    """Long-term memory: store every user message, retrieve the most relevant ones.
    Combined with a short window for recent context (the common production pattern)."""

    def __init__(self, path=None, k_recent=2, top_k=2):
        self.path, self.top_k = path, top_k
        self.window = WindowMemory(k_recent)
        self.items = []
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    self.items = json.load(f)
            except FileNotFoundError:
                pass

    def add(self, user, assistant):
        self.window.add(user, assistant)
        self.items.append(user)
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, indent=2)

    def retrieve(self, query):
        q = embed(query)
        scored = sorted(((cosine(q, embed(t)), t) for t in self.items), reverse=True)
        return [t for score, t in scored[: self.top_k] if score > 0]

    def context(self, query):
        hits = self.retrieve(query)
        extra = ("Relevant memories about the user:\n" + "\n".join(f"- {h}" for h in hits)) if hits else ""
        return extra, self.window.context(query)[1]
