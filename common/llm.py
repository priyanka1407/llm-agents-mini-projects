"""
Tiny wrapper around the Claude API that every project in this repo shares.

Two modes:
  REAL  - calls Claude (needs `pip install anthropic` and ANTHROPIC_API_KEY)
  MOCK  - a scripted fake "brain" per project, so you can run and study the
          agent loop offline with no key and no cost.

Mock responses have the same shape as real SDK responses
(.stop_reason, .content blocks with .type / .text / .name / .input / .id),
so the agent code is identical in both modes.
"""
import os
from types import SimpleNamespace as NS

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")


def use_mock() -> bool:
    if os.getenv("MOCK_LLM") == "1":
        return True
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return True
    return not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


class LLM:
    def __init__(self, mock_brain=None):
        self.mock = use_mock()
        self.mock_brain = mock_brain
        if not self.mock:
            import anthropic
            self.client = anthropic.Anthropic()
        print(f"[LLM mode: {'MOCK (offline)' if self.mock else 'REAL - ' + MODEL}]")

    def create(self, system, messages, tools=None, max_tokens=8000):
        if self.mock:
            return self.mock_brain(system, messages, tools)
        kwargs = dict(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            # If the main model declines a request, the API retries on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if tools:
            kwargs["tools"] = tools
        return self.client.beta.messages.create(**kwargs)


def text_of(response) -> str:
    """Join all text blocks of a response (skips thinking / tool_use blocks)."""
    if response.stop_reason == "refusal":
        return "[The model declined this request.]"
    return "".join(b.text for b in response.content if b.type == "text").strip()


def run_tool_loop(llm, system, task, tools, tool_funcs, max_steps=6, verbose=True):
    """The standard agent loop: think -> call tools -> observe -> repeat -> answer."""
    messages = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        response = llm.create(system, messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return text_of(response)
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = str(tool_funcs[block.name](**block.input)), False
            except Exception as e:  # tell the model, don't crash
                output, is_error = f"Error: {e}", True
            if verbose:
                print(f"    [step {step}] {block.name}({block.input}) -> {output[:80]}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return "Stopped: reached max steps."


# ---------- helpers for building mock responses ----------

def mock_text(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def mock_tool_calls(calls):
    """calls = [(tool_name, {input}), ...]"""
    blocks = [NS(type="tool_use", id=f"toolu_mock_{i}", name=name, input=inp)
              for i, (name, inp) in enumerate(calls)]
    return NS(stop_reason="tool_use", content=blocks)


def last_tool_results(messages):
    """In mock mode: return {tool_name: output} for the latest tool results, or None."""
    last = messages[-1]["content"]
    if not isinstance(last, list):
        return None
    names = {b.id: b.name for b in messages[-2]["content"] if getattr(b, "type", "") == "tool_use"}
    return {names[r["tool_use_id"]]: r["content"] for r in last}
