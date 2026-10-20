# 02 · Multi-Agent System (Planner → Researcher → Writer ⇄ Critic)

**One line:** split a big job into specialist agents, each with its own role, tools and small context, coordinated by an orchestrator.

## Flow
```
             topic
               │
          ┌────▼────┐
          │ PLANNER │  → 3 sub-questions
          └────┬────┘
     ┌─────────┼─────────┐
┌────▼─────┐┌──▼───────┐┌▼─────────┐
│RESEARCHER││RESEARCHER││RESEARCHER│  (each uses search_knowledge tool)
└────┬─────┘└──┬───────┘└┬─────────┘
     └─────────┼─────────┘  notes
          ┌────▼───┐
          │ WRITER │◄──────────┐
          └────┬───┘           │ REVISE: feedback
          ┌────▼───┐           │
          │ CRITIC │───────────┘
          └────┬───┘
            APPROVED → final article
```

## What makes it "multi-agent"
| Idea | In the code |
|---|---|
| **Agent = role + tools + context** | `Agent(name, system, tools)`: same LLM, different system prompt |
| **Orchestrator** | `orchestrate()`: plain Python decides who runs next (deterministic, easy to debug) |
| **Tool-using worker** | Researcher reuses the single-agent loop (`common/llm.py → run_tool_loop`) |
| **Reflection / critic loop** | Writer ⇄ Critic, capped by `max_revisions` |
| **Context isolation** | Each researcher only sees its own question, so prompts stay small and focused |
| **Parallelism** | Research calls are independent and could run concurrently |

## Patterns you can name in interviews
- **Orchestrator–worker (supervisor)**: this project.
- **Sequential pipeline**: A → B → C with no loop.
- **Reflection / evaluator–optimizer**: writer + critic.
- **Hierarchical**: a supervisor of supervisors.
- **Peer handoff (swarm)**: agents pass control to each other directly.

## When NOT to use multi-agent
More agents = more tokens, latency and failure points. Use them when sub-tasks are **independent** (parallel research) or need **different tools/expertise**. Otherwise one agent with good tools is simpler.

## Run
```bash
python orchestrator.py                    # mock mode (offline)
python orchestrator.py "Vector databases"
```

## Interview line
> "In my multi-agent demo an orchestrator coordinates a planner, tool-using researchers, a writer and a critic. Each agent is the same LLM with its own role prompt, tools and isolated context. The critic loop gives self-correction, and I cap revisions so it can't loop forever. I only reach for multi-agent when sub-tasks are independent or need different tools. Otherwise one well-tooled agent is cheaper and easier to debug."
