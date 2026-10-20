# LLM Agents: Mini Projects

Small, readable projects that show how LLM agents work under the hood. No frameworks: just Python and the Claude API, so every step of the loop is visible.

| # | Project | Concepts |
|---|---|---|
| 01 | [Single agent](01-single-agent) | Tool use, ReAct loop, parallel tool calls, error handling, step limits |
| 02 | [Multi-agent](02-multi-agent) | Orchestrator–worker, role prompts, critic/reflection loop, context isolation |
| 03 | [Memory techniques](03-memory-techniques) | Buffer, sliding window, summary memory, vector long-term memory |

## Runs offline by default
Every project has a **mock mode**: a small scripted "fake LLM" that returns responses in the same shape as the real API. You can run and study each loop with **no API key and no cost**.

```bash
python 01-single-agent/agent.py
python 02-multi-agent/orchestrator.py
python 03-memory-techniques/demo.py
```

## Use real Claude
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python 01-single-agent/agent.py "What is 15% of 2400 and the weather in Dubai?"
```
- Default model: `claude-opus-5-5` (change it with `CLAUDE_MODEL`).
- Server-side refusal fallback is enabled (`fallbacks="default"`).
- Set `MOCK_LLM=1` to force mock mode.

## Structure
```
common/llm.py          shared client: real/mock switch, text_of(), run_tool_loop()
01-single-agent/       agent.py + README
02-multi-agent/        orchestrator.py + README
03-memory-techniques/  memory.py, demo.py + README
```

Related: [ml-concepts-from-scratch](https://github.com/priyanka1407/ml-concepts-from-scratch): the ML foundations (regression, backprop, self-attention).
