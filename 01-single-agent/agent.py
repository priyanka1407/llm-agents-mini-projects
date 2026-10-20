"""
Single agent with tools - the ReAct / tool-use loop written out by hand.

The agent can:
  - calculator   : do exact math (LLMs are bad at arithmetic)
  - get_weather  : look up (fake) weather for a city
  - search_notes : search a tiny company handbook

Run:  python agent.py            (mock mode if no API key)
      python agent.py "What is 15% of 2400?"
"""
import ast
import operator
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llm import LLM, text_of, mock_text, mock_tool_calls, last_tool_results  # noqa: E402

# ---------------- 1. Tools (plain Python functions) ----------------

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.USub: operator.neg}


def calculator(expression: str) -> str:
    """Safe math: walks the syntax tree instead of using eval()."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** % are allowed")
    return str(round(ev(ast.parse(expression, mode="eval").body), 4))


WEATHER = {"chennai": "33 C, humid, light rain in the evening",
           "bangalore": "26 C, cloudy",
           "london": "12 C, drizzle",
           "dubai": "38 C, sunny"}


def get_weather(city: str) -> str:
    return WEATHER.get(city.lower(), f"No weather data for {city}")


NOTES = ["Leave policy: employees get 24 days of paid leave per year.",
         "Work from home: allowed 2 days per week with manager approval.",
         "Expense policy: claims must be submitted within 30 days with receipts."]


STOPWORDS = {"what", "how", "many", "do", "i", "the", "a", "is", "in", "and", "get", "s", "per", "with", "my", "days"}


def search_notes(query: str) -> str:
    words = set(re.findall(r"\w+", query.lower())) - STOPWORDS
    hits = [n for n in NOTES if words & set(re.findall(r"\w+", n.lower())) - STOPWORDS]
    return " | ".join(hits) or "No matching notes."


TOOL_FUNCS = {"calculator": calculator, "get_weather": get_weather, "search_notes": search_notes}

# ---------------- 2. Tool descriptions the LLM reads ----------------

TOOLS = [
    {"name": "calculator",
     "description": "Evaluate an arithmetic expression, e.g. '2400 * 0.15'. Use this for ANY math.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]},
     "strict": True},
    {"name": "get_weather",
     "description": "Get today's weather for a city.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "properties": {"city": {"type": "string"}},
                      "required": ["city"]},
     "strict": True},
    {"name": "search_notes",
     "description": "Search the company handbook (leave, work from home, expenses).",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]},
     "strict": True},
]

SYSTEM = ("You are a helpful assistant. Use the tools when they help; you may call several "
          "tools at once. Answer briefly and only with facts from tool results.")

# ---------------- 3. The agent loop ----------------


def run_agent(llm, question, max_steps=6):
    messages = [{"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        # THINK: the model decides to answer or to call tools
        response = llm.create(SYSTEM, messages, TOOLS)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":       # no tools needed -> final answer
            return text_of(response)

        # ACT: run every tool the model asked for
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [step {step}] ACT     {block.name}({block.input})")
            try:
                output, is_error = TOOL_FUNCS[block.name](**block.input), False
            except Exception as e:
                output, is_error = f"Error: {e}", True
            print(f"  [step {step}] OBSERVE {output}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output, "is_error": is_error})

        # OBSERVE: send all results back in ONE user message, then loop
        messages.append({"role": "user", "content": results})
    return "Stopped: reached max steps."

# ---------------- 4. Offline mock brain (pretends to be the LLM) ----------------


def mock_brain(system, messages, tools):
    results = last_tool_results(messages)
    if results:                                           # we already have observations
        return mock_text("Here is what I found: " + "; ".join(results.values()))
    q = messages[-1]["content"].lower()
    calls = []
    math = re.findall(r"[\d.]+\s*%\s*of\s*[\d.]+|[\d.(][\d.\s+\-*/()]*[+\-*/][\d.\s+\-*/()]*[\d)]", q)
    if math:
        expr = math[0]
        pct = re.match(r"([\d.]+)\s*%\s*of\s*([\d.]+)", expr)
        calls.append(("calculator", {"expression": f"{pct[1]} / 100 * {pct[2]}" if pct else expr}))
    for city in WEATHER:
        if city in q:
            calls.append(("get_weather", {"city": city.title()}))
    if any(w in q for w in ["leave", "home", "expense", "policy"]):
        calls.append(("search_notes", {"query": q}))
    if not calls:
        return mock_text("I can help with math, weather and the company handbook.")
    return mock_tool_calls(calls)


if __name__ == "__main__":
    llm = LLM(mock_brain)
    questions = sys.argv[1:] or [
        "What is 15% of 2400?",
        "What's the weather in Chennai, and how many leave days do I get?",
        "Tell me a joke.",
    ]
    for q in questions:
        print(f"\nUSER: {q}")
        print(f"AGENT: {run_agent(llm, q)}")
