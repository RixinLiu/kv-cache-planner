"""The built-in cases: a model's config.json and a deployment, each there to test one rule."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from contract import CASES, CaseOutput


@dataclass(frozen=True)
class CaseSpec:
    config: dict[str, Any]  # the model's config.json
    gpu_memory: str
    reserved_memory: str
    context_length: int
    kv_cache_dtype: str
    tests: str  # what the case is there to test


LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_act": "silu", "rms_norm_eps": 1e-05}

SPECS = {
    1: CaseSpec(
        config={
            **LLAMA,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        },
        gpu_memory="80 GiB",
        reserved_memory="4 GiB",
        context_length=4096,
        kv_cache_dtype="auto",
        tests="grouped-query attention: 8 KV heads for 32 query heads",
    ),
    2: CaseSpec(
        config={
            **LLAMA,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
        },
        gpu_memory="24 GB",
        reserved_memory="1 GB",
        context_length=2048,
        kv_cache_dtype="auto",
        tests="no num_key_value_heads (every head has its own KV) and decimal GB",
    ),
    3: CaseSpec(
        config={
            "architectures": ["GemmaForCausalLM"],
            "model_type": "gemma",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-06,
            "attention_bias": False,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
        },
        gpu_memory="16 GiB",
        reserved_memory="1536 MiB",
        context_length=4096,
        kv_cache_dtype="fp8",
        tests="multi-query attention (one KV head), tied embeddings and an fp8 KV cache",
    ),
    4: CaseSpec(
        config={
            **LLAMA,
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "attention_bias": False,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        },
        gpu_memory="80 GiB",
        reserved_memory="2 GiB",
        context_length=4096,
        kv_cache_dtype="auto",
        tests="weights larger than the GPU, so usable memory is negative and no sequence fits",
    ),
    5: CaseSpec(
        config={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-06,
            "attention_bias": True,
            "tie_word_embeddings": True,
            "torch_dtype": "float32",
        },
        gpu_memory="48 GB",
        reserved_memory="2 GB",
        context_length=16384,
        kv_cache_dtype="float16",
        tests="a head_dim other than hidden_size / heads, attention biases, float32 weights and a float16 KV cache",
    ),
    6: CaseSpec(
        config={
            **LLAMA,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 23,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 50257,
            "max_position_embeddings": 16384,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
        },
        gpu_memory="24 GiB",
        reserved_memory="1735 MiB",
        context_length=8192,
        kv_cache_dtype="auto",
        tests="usable memory exactly 16 sequences' KV caches, so the 16th fits with nothing to spare",
    ),
}


def case(number: int) -> CaseOutput:
    """Built-in case ``number``."""
    if number not in SPECS:
        raise ValueError(f"case must be 1 to {CASES}, not {number}")
    return build_case(number, SPECS[number])


def build_case(number: int, spec: CaseSpec) -> CaseOutput:
    return CaseOutput(
        case=number,
        config_json=json.dumps(spec.config, indent=2),
        gpu_memory=spec.gpu_memory,
        reserved_memory=spec.reserved_memory,
        context_length=spec.context_length,
        kv_cache_dtype=spec.kv_cache_dtype,
    )
