"""The plan a case asks for, by the rules the task's instructions give. The agent works it out with calculate."""

from __future__ import annotations

import json

from contract import CaseOutput, PlanOutput

UNITS = {"GiB": 2**30, "GB": 10**9, "MiB": 2**20, "MB": 10**6}
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "fp8": 1}


def memory_bytes(text: str) -> int:
    """``80 GiB`` as bytes: a whole number, one space, and a unit of UNITS."""
    number, unit = text.split(" ")
    if not number.isdigit() or unit not in UNITS:
        raise ValueError(f"not a whole number and a unit of {', '.join(UNITS)}: {text!r}")
    return int(number) * UNITS[unit]


def plan(case: CaseOutput) -> PlanOutput:
    config = json.loads(case.config_json)
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", hidden // heads)
    layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    attention = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    bias = (heads + 2 * kv_heads) * head_dim if config.get("attention_bias", False) else 0
    layer = attention + bias + 3 * hidden * config["intermediate_size"] + 2 * hidden
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab * hidden
    params = layers * layer + hidden + vocab * hidden + lm_head

    weight_bytes = params * DTYPE_BYTES[config["torch_dtype"]]
    kv_dtype = config["torch_dtype"] if case.kv_cache_dtype == "auto" else case.kv_cache_dtype
    per_token = 2 * layers * kv_heads * head_dim * DTYPE_BYTES[kv_dtype]
    per_sequence = per_token * case.context_length
    usable = memory_bytes(case.gpu_memory) - memory_bytes(case.reserved_memory) - weight_bytes
    return PlanOutput(
        param_count=params,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=per_token,
        kv_bytes_per_sequence=per_sequence,
        usable_bytes=usable,
        max_batch=usable // per_sequence if usable > 0 else 0,
    )
