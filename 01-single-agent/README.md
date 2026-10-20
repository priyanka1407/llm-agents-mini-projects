# 01 · Single Agent with Tools (ReAct loop)

**One line:** an LLM in a loop that can call tools until it has enough to answer.

## The loop
```
USER question
   │
   ▼
THINK   LLM reads question + tool descriptions
   │      ├─ "I can answer"      → final answer ✅
   │      └─ "I need a tool"     → tool_use block(s)
   ▼
ACT     our Python code runs the tool(s)
   ▼
OBSERVE tool_result goes back to the LLM
   └──────── repeat (max_steps guard) ────────┘
```

## Example run
```
USER: What's the weather in Chennai, and how many leave days do I get?
  [step 1] ACT     get_weather({'city': 'Chennai'})
  [step 1] ACT     search_notes({'query': '... leave days ...'})   <- 2 tools in parallel
  [step 1] OBSERVE 33 C, humid ...
  [step 1] OBSERVE Leave policy: employees get 24 days ...
AGENT: It's 33 C and humid in Chennai, and you get 24 days of paid leave.
```

## Key code ideas (`agent.py`)
| Piece | What it does |
|---|---|
| `TOOLS` | JSON schema + description per tool. The **description is the prompt** the LLM uses to decide. |
| `strict: True` | The model's tool arguments always match the schema. |
| `stop_reason == "tool_use"` | The model wants a tool → run it. Anything else → final answer. |
| All `tool_result`s in **one** user message | Required for parallel tool calls. |
| `is_error: True` | Tool failures go back to the model so it can recover, instead of crashing. |
| `max_steps` | Stops infinite loops. |
| Safe `calculator` | Walks the syntax tree instead of `eval()`, so no code injection. |

## Run
```bash
python agent.py                         # mock mode (offline, no key)
python agent.py "What is 15% of 2400?"
# real Claude:
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py
```

## Interview line
> "A single agent is an LLM in a loop: it reads the tool schemas, emits tool calls, my code executes them and returns results, and it repeats until the model gives a final answer. I add a step limit, return tool errors to the model instead of crashing, and keep tools narrow and well-described, because the description is what the model reasons over."
