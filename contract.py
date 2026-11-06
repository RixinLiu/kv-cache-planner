"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    READ_CASE = "read_case"
    CALCULATE = "calculate"
    CASE_RECORD = "case_record"


class Task(Enum):
    PLAN_KV_CACHE = "plan-kv-cache"


# How many built-in cases there are: a case is numbered 1 to CASES.
CASES = 6
# The longest expression calculate takes, in characters.
MAX_EXPRESSION = 500

CASE = "The case's number, 1 to 6"


@dataclass
class CaseInput(Output):
    case: Annotated[int, CASE]


@dataclass
class CaseOutput(Output):
    case: Annotated[int, CASE]
    config_json: Annotated[str, "The model's config.json, as Hugging Face publishes it"]
    gpu_memory: Annotated[str, "The GPU's memory, a whole number and a unit: GiB, GB, MiB or MB, e.g. 80 GiB"]
    reserved_memory: Annotated[
        str, "Memory kept back from the model and its KV cache (CUDA context, activations), a whole number and a unit"
    ]
    context_length: Annotated[int, "How many tokens each sequence's KV cache must hold"]
    kv_cache_dtype: Annotated[str, "The KV cache's data type: auto (the weights' dtype), fp8, bfloat16 or float16"]


@dataclass
class CalculateInput(Output):
    expression: Annotated[
        str, "Integers combined with + - * // ** and parentheses, e.g. 2 * 32 * 8 * 128 * 2; no names, / or decimals"
    ]


@dataclass
class CalculateOutput(Output):
    expression: Annotated[str, "The expression, as given without surrounding spaces"]
    value: Annotated[int, "Its exact value; // rounds down, as Python's does"]


@dataclass
class PlanOutput(Output):
    param_count: Annotated[int, "The model's parameters"]
    weight_bytes: Annotated[int, "The weights' size: parameters times the bytes of the weights' dtype"]
    kv_bytes_per_token: Annotated[int, "The KV cache one token takes, keys and values over every layer"]
    kv_bytes_per_sequence: Annotated[int, "The KV cache one sequence of context_length tokens takes"]
    usable_bytes: Annotated[int, "GPU memory less reserved memory and the weights; negative when the weights do not fit"]
    max_batch: Annotated[int, "How many whole sequences' KV caches fit in usable_bytes; 0 when usable_bytes is not positive"]


TOOLS = {
    Tool.READ_CASE: (CaseInput, CaseOutput),
    Tool.CALCULATE: (CalculateInput, CalculateOutput),
    Tool.CASE_RECORD: (CaseInput, CaseOutput),
}
TASKS = {Task.PLAN_KV_CACHE: (CaseInput, PlanOutput)}
