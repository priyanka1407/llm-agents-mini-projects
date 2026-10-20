"""
Same conversation, four memory strategies. Which one still remembers
the user's name and city at the end, and how big is the prompt?

Run:  python demo.py      (mock mode if no API key)
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llm import LLM, text_of, mock_text  # noqa: E402
from memory import BufferMemory, WindowMemory, SummaryMemory, VectorMemory  # noqa: E402

BASE_SYSTEM = "You are a friendly assistant. Answer in one or two sentences."

CONVERSATION = [
    "Hi, my name is Priya and I live in Chennai.",
    "I work as a QA engineer and I'm moving into AI engineering.",
    "Tell me a fun fact about space.",
    "Suggest one book about machine learning.",
    "Give me one tip for a technical interview.",
]
FINAL_QUESTION = "What is my name and where do I live?"


def chat(llm, memory, user_msg):
    extra, history = memory.context(user_msg)
    system = BASE_SYSTEM + ("\n\n" + extra if extra else "")
    messages = history + [{"role": "user", "content": user_msg}]
    reply = text_of(llm.create(system, messages))
    memory.add(user_msg, reply)
    prompt_words = len((system + " ".join(m["content"] for m in messages)).split())
    return reply, prompt_words


def mock_brain(system, messages, tools):
    """Pretend LLM: can only 'know' facts that are actually inside the prompt."""
    prompt = system + "\n" + "\n".join(m["content"] for m in messages)
    if system.startswith("ROLE: summarizer"):
        request = messages[-1]["content"]
        old = request.split("Current summary:\n")[1].split("\n\nNew lines:")[0].replace("(empty)", "")
        new = re.findall(r"(my name is \w+|i live in \w+|i work as an? [\w ]+?)(?:[.,]| and)",
                         request.split("New lines:")[1], re.I)
        facts = [f.strip() for f in old.split(";") if f.strip()] + [f"user said '{f}'" for f in new]
        return mock_text("; ".join(dict.fromkeys(facts)))
    question = messages[-1]["content"].lower()
    if "my name" in question:
        name = re.search(r"my name is (\w+)", prompt, re.I)
        city = re.search(r"live in (\w+)", prompt, re.I)
        if name or city:
            return mock_text(f"Your name is {name[1] if name else '?'} and you live in {city[1] if city else '?'}.")
        return mock_text("Sorry, I don't know your name or city.")
    return mock_text(f"(reply to: {messages[-1]['content'][:40]})")


if __name__ == "__main__":
    llm = LLM(mock_brain)
    store = os.path.join(tempfile.gettempdir(), "vector_memory_demo.json")
    if os.path.exists(store):
        os.remove(store)

    strategies = {
        "1. Buffer (everything)": BufferMemory(),
        "2. Window (last 2 turns)": WindowMemory(k=2),
        "3. Summary + last 2 turns": SummaryMemory(llm, keep_last=2),
        "4. Vector long-term + last 2": VectorMemory(path=store, k_recent=2),
    }

    for name, memory in strategies.items():
        for msg in CONVERSATION:
            chat(llm, memory, msg)
        answer, size = chat(llm, memory, FINAL_QUESTION)
        print(f"\n{name}\n   prompt size : ~{size} words\n   answer      : {answer}")

    # Long-term memory survives a "restart": a brand-new object loads the saved store
    fresh = VectorMemory(path=store)
    answer, _ = chat(llm, fresh, FINAL_QUESTION)
    print(f"\n5. New session, vector memory loaded from disk\n   answer      : {answer}")
