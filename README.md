# KV Cache Planner

A `nanoagent.plugin/v0001` plugin that asks an agent to size an LLM
deployment from a model's `config.json`: its parameters, its weights in bytes,
its KV cache per token and per sequence, and how many sequences fit on a GPU.
The verifier works out the same six integers on its own and checks each
exactly.

## Tools

| Tool | Does | Code |
|---|---|---|
| `read_case` | Reads a case: a model's `config.json`, the GPU's memory, the memory kept back, the context length and the KV cache's dtype | `src/plugin/case.py` |
| `calculate` | Works out an integer expression exactly, with `+ - * // **` and parentheses, as Python does; refuses names, `/`, decimals and steps past 10**30 | `src/plugin/calculate.py` |
| `case_record` | Reads a case as `read_case` does: the `plan-kv-cache` task's reference, pinned for the verifier where the agent never sees it | `src/plugin/case.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `plan-kv-cache` | Works out a model's size, its KV cache per token and per sequence, and how many sequences fit on a GPU | `agent`, with `read_case` and `calculate`; reference `case_record` |

The task's input is a case number, 1 to 6 (`src/plugin/cases.py`). Cases 1
to 4 are the published configs of Llama-3-8B, Llama-2-7B, Gemma-2B and
Llama-3-70B, and their parameter counts match the published ones. Each case is
there to test one rule:

| Case | Model | GPU, kept back, context, KV dtype | Tests |
|---|---|---|---|
| 1 | Llama-3-8B, bfloat16 | 80 GiB, 4 GiB, 4096, auto | grouped-query attention: 8 KV heads for 32 query heads |
| 2 | Llama-2-7B, float16 | 24 GB, 1 GB, 2048, auto | no `num_key_value_heads`, so every head has its own KV; decimal GB |
| 3 | Gemma-2B, bfloat16 | 16 GiB, 1536 MiB, 4096, fp8 | one KV head, tied embeddings, an fp8 KV cache |
| 4 | Llama-3-70B, bfloat16 | 80 GiB, 2 GiB, 4096, auto | weights larger than the GPU: usable memory is negative and `max_batch` is 0 |
| 5 | a Qwen2-style 2.3B, float32 | 48 GB, 2 GB, 16384, float16 | `head_dim` 128 where `hidden_size / heads` is 80, attention biases, a KV dtype other than the weights' |
| 6 | a 5.8B with GPT-2's vocabulary, bfloat16 | 24 GiB, 1735 MiB, 8192, auto | usable memory is exactly 16 sequences' KV caches, so the 16th fits |

The instructions give every rule: the config's defaults (no
`num_key_value_heads` means one KV head per query head, no `head_dim` means
`hidden_size // num_attention_heads`), each layer's parameters, the dtypes'
bytes, the memory units (GiB is 1024**3 bytes, GB is 1000**3), and
`max_batch` rounded down and never below 0. The agent reads the config,
applies the rules and does every product and quotient with `calculate`.

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs (none)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS), the tools, plan.py and the cases
├── tests/           # test_verifier.py (right, wrong, missing, malformed and boundary plans) and test_plan.py
├── ui/              # outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, with its own planner and expression parser
```

`verifier/` imports only the standard library and `contract`. For the task it
takes the pinned `case_record` output, checks the case can be planned, works
out the six integers with its own planner (`verifier/plan.py`, written apart
from `src/plugin/plan.py`), and compares each field exactly. A wrong plan's
message names the first wrong field with the value expected, and how many
others are wrong; the score is the share of fields that are right. For each
`calculate` call it evaluates the expression again with its own parser
(`verifier/calculate.py`, since the verifier may not import `ast`); for each
case it reads, it checks the config and the memory sizes can be planned.

`tests/test_plan.py` checks the two planners agree on 500 random configs and
the two calculators agree with Python on 300 random expressions.
`tests/test_verifier.py` checks the verifier approves each case's plan and
refuses plans made with one mistake each, and that each mistake changes the
answer of at least one case: GQA ignored, `head_dim` taken as
`hidden_size // heads`, an MLP that is not gated, norms, the embeddings or
attention biases forgotten, tied embeddings counted the other way, the KV
dtype taken from the weights, the factor of 2 for keys and values dropped,
`max_position_embeddings` taken as the context, GB read as GiB, reserved
memory ignored, a negative `max_batch`, `max_batch` rounded up or with one
byte kept spare, any field off by one, and the per-token and per-sequence
sizes swapped.

## Develop and run

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`;
this plugin writes nothing.
