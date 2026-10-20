# 03 · Memory Techniques for LLM Agents

**One line:** LLMs are stateless. "Memory" means choosing what past information goes back into the next prompt.

## The test
Five chat turns: the name and city come up in turn 1. Then we ask *"What is my name and where do I live?"*

| Strategy | Prompt size | Remembers? | Idea |
|---|---|---|---|
| 1. **Buffer** | ~110 words (grows forever) | ✅ | Send the whole history |
| 2. **Window** (last 2 turns) | ~51 words | ❌ forgot | Cheap, but old facts drop off |
| 3. **Summary** + last 2 | ~75 words | ✅ | LLM folds old turns into a running summary |
| 4. **Vector** long-term + last 2 | ~67 words | ✅ | Store messages, retrieve the most similar ones |
| 5. Vector, **new session** | — | ✅ | Saved to disk, so it survives a restart |

## How each one works
```
Buffer  : [t1 t2 t3 t4 t5] + question
Window  :          [t4 t5] + question
Summary : "User is Priya, lives in Chennai" + [t4 t5] + question
Vector  : search(question) -> "my name is Priya ... Chennai" + [t4 t5] + question
```

## Short-term vs long-term memory
- **Short-term (working) memory**: the current conversation: buffer, window, summary.
- **Long-term memory**: facts that survive across sessions: vector store, database, user profile.
- Production agents usually **combine** them: a recent window + retrieved long-term memories + a summary.

## Code map
| File | What |
|---|---|
| `memory.py` | `BufferMemory`, `WindowMemory`, `SummaryMemory`, `VectorMemory` (all share `add()` / `context()`) |
| `demo.py` | Runs the same conversation through every strategy and compares |

`VectorMemory` uses a toy word-count embedding + cosine similarity so it runs without extra installs. In production, swap in a real embedding model and a vector DB (Qdrant / Chroma / Weaviate).

## Run
```bash
python demo.py          # mock mode (offline)
```

## Interview line
> "Because LLMs are stateless, memory is context management. A buffer is simplest but grows without limit. A sliding window is cheap but forgets. Summary memory compresses old turns with the LLM. For long-term memory I embed and store facts in a vector DB and retrieve the relevant ones per query. In practice I combine a recent window, a running summary and retrieved long-term memories."
