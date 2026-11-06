"""The verifier's own planner, written apart from the plugin's so a slip in one shows against the other."""

import json
import re

MEMORY = re.compile(r"(\d+) (GiB|GB|MiB|MB)")
UNIT_BYTES = {"GiB": 1024**3, "GB": 1000**3, "MiB": 1024**2, "MB": 1000**2}
WEIGHT_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
KV_BYTES = {"bfloat16": 2, "float16": 2, "fp8": 1}
KV_DTYPES = ("auto", *KV_BYTES)
REQUIRED = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
# The plan's fields, in the order the task's output declares them.
FIELDS = ("param_count", "weight_bytes", "kv_bytes_per_token", "kv_bytes_per_sequence", "usable_bytes", "max_batch")


def memory(text):
    """Bytes of a memory size such as ``80 GiB``, or None when it is not one."""
    match = MEMORY.fullmatch(text)
    if match is None:
        return None
    return int(match.group(1)) * UNIT_BYTES[match.group(2)]


def positive(value):
    return type(value) is int and value > 0


def config_problem(config):
    """What keeps a config.json from being planned, or None."""
    if type(config) is not dict:
        return "config.json is not an object"
    for key in REQUIRED:
        if not positive(config.get(key)):
            return f"config.json's {key} is not a positive integer"
    if config.get("torch_dtype") not in WEIGHT_BYTES:
        return f"config.json's torch_dtype is not one of {', '.join(WEIGHT_BYTES)}"
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    if not positive(kv_heads) or heads % kv_heads:
        return "num_key_value_heads must be a positive integer that divides num_attention_heads"
    if "head_dim" in config and not positive(config["head_dim"]):
        return "head_dim is not a positive integer"
    if "head_dim" not in config and config["hidden_size"] % heads:
        return "hidden_size does not divide by num_attention_heads, and there is no head_dim"
    for key in ("attention_bias", "tie_word_embeddings"):
        if type(config.get(key, False)) is not bool:
            return f"config.json's {key} is not true or false"
    return None


def case_problem(case):
    """What keeps a contract CaseOutput from being planned, or None."""
    try:
        config = json.loads(case.config_json)
    except ValueError:
        return "config_json is not JSON"
    problem = config_problem(config)
    if problem:
        return problem
    for name, text in (("gpu_memory", case.gpu_memory), ("reserved_memory", case.reserved_memory)):
        if memory(text) is None:
            return f"{name} {text!r} is not a whole number and a unit (GiB, GB, MiB or MB)"
    if case.context_length <= 0:
        return "context_length must be positive"
    if case.kv_cache_dtype not in KV_DTYPES:
        return f"kv_cache_dtype must be one of {', '.join(KV_DTYPES)}"
    return None


def expected_plan(case):
    """``case``: a contract CaseOutput that case_problem passes. The plan's fields by name."""
    config = json.loads(case.config_json)
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", hidden // heads)
    vocab = config["vocab_size"]

    # One decoder layer: the q, k, v and o projections, their biases, a gated MLP (gate, up, down), two norms.
    q_proj = hidden * heads * head_dim
    k_proj = v_proj = hidden * kv_heads * head_dim
    o_proj = heads * head_dim * hidden
    biases = heads * head_dim + 2 * kv_heads * head_dim if config.get("attention_bias") is True else 0
    mlp = 3 * hidden * config["intermediate_size"]
    norms = hidden + hidden
    per_layer = q_proj + k_proj + v_proj + o_proj + biases + mlp + norms
    embeddings = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings") is True else vocab * hidden
    params = layers * per_layer + hidden + embeddings + lm_head

    weight_bytes = WEIGHT_BYTES[config["torch_dtype"]] * params
    kv_bytes = WEIGHT_BYTES[config["torch_dtype"]] if case.kv_cache_dtype == "auto" else KV_BYTES[case.kv_cache_dtype]
    per_token = layers * 2 * kv_heads * head_dim * kv_bytes  # a key and a value per KV head in every layer
    per_sequence = case.context_length * per_token
    usable = memory(case.gpu_memory) - memory(case.reserved_memory) - weight_bytes
    max_batch = 0
    if usable > 0:
        max_batch = usable // per_sequence
    values = (params, weight_bytes, per_token, per_sequence, usable, max_batch)
    return dict(zip(FIELDS, values, strict=True))
