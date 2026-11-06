from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import CASES, TASKS, TOOLS, CaseOutput, PlanOutput, Task, Tool
from plugin import cases
from plugin.plan import plan

TASK = Task.PLAN_KV_CACHE.value
FIELDS = ["param_count", "weight_bytes", "kv_bytes_per_token", "kv_bytes_per_sequence", "usable_bytes", "max_batch"]
BINARY = {"GiB": 2**30, "GB": 2**30, "MiB": 2**20, "MB": 2**20}
DECIMAL = {"GiB": 2**30, "GB": 10**9, "MiB": 2**20, "MB": 10**6}
BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "fp8": 1}


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(case: int, output: dict, reference: CaseOutput | None, workspace: Path) -> dict:
    given, made = TASKS[Task.PLAN_KV_CACHE]
    pinned = None if reference is None else build_output(plain_output(reference), CaseOutput)
    return verifier.verify_task(TASK, build_output({"case": case}, given), build_output(output, made), pinned, str(workspace))


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def verify_output(tool: Tool, given: dict, made: dict) -> dict:
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(tool.value, build_output(given, given_class), build_output(made, made_class), ".")


def plan_with(case: CaseOutput, mistake: str | None = None) -> dict:
    """The case's plan worked out with one mistake an agent could make, or none."""
    config = json.loads(case.config_json)
    hidden, layers, heads = config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"]
    vocab, inter = config["vocab_size"], config["intermediate_size"]
    kv_heads = heads if mistake == "GQA ignored" else config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", hidden // heads)
    if mistake == "head_dim taken as hidden // heads":
        head_dim = hidden // heads
    layer = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    layer += (2 if mistake == "MLP not gated" else 3) * hidden * inter
    layer += 0 if mistake == "layer norms forgotten" else 2 * hidden
    if config.get("attention_bias", False) and mistake != "attention bias forgotten":
        layer += (heads + 2 * kv_heads) * head_dim
    tied = config.get("tie_word_embeddings", False) != (mistake == "tied lm_head counted the other way")
    params = layers * layer + (0 if mistake == "final norm forgotten" else hidden)
    params += (0 if mistake == "embeddings forgotten" else vocab * hidden) + (0 if tied else vocab * hidden)
    weight_bytes = params * BYTES[config["torch_dtype"]]
    auto = case.kv_cache_dtype == "auto" or mistake == "KV dtype taken from the weights"
    kv_dtype = config["torch_dtype"] if auto else case.kv_cache_dtype
    per_token = (1 if mistake == "K and V factor of 2 dropped" else 2) * layers * kv_heads * head_dim * BYTES[kv_dtype]
    context = config["max_position_embeddings"] if mistake == "max_position_embeddings as context" else case.context_length
    per_sequence = per_token * context
    units = BINARY if mistake == "GB read as GiB" else DECIMAL

    def size(text: str) -> int:
        number, unit = text.split(" ")
        return int(number) * units[unit]

    reserved = 0 if mistake == "reserved memory ignored" else size(case.reserved_memory)
    usable = size(case.gpu_memory) - reserved - weight_bytes
    if mistake == "negative max_batch":
        max_batch = usable // per_sequence
    elif usable <= 0:
        max_batch = 0
    elif mistake == "rounded up":
        max_batch = -(-usable // per_sequence)
    elif mistake == "one byte kept spare":
        max_batch = (usable - 1) // per_sequence
    else:
        max_batch = usable // per_sequence
    return dict(zip(FIELDS, [params, weight_bytes, per_token, per_sequence, usable, max_batch], strict=True))


MISTAKES = [
    "GQA ignored",
    "head_dim taken as hidden // heads",
    "MLP not gated",
    "layer norms forgotten",
    "final norm forgotten",
    "embeddings forgotten",
    "attention bias forgotten",
    "tied lm_head counted the other way",
    "KV dtype taken from the weights",
    "K and V factor of 2 dropped",
    "max_position_embeddings as context",
    "GB read as GiB",
    "reserved memory ignored",
    "negative max_batch",
    "rounded up",
    "one byte kept spare",
]


def mutants(case: CaseOutput) -> dict[str, dict]:
    """Wrong answers, each differing from the right one in one way; only those that change the answer."""
    right = plan_with(case)
    found = {mistake: plan_with(case, mistake) for mistake in MISTAKES}
    for field in FIELDS:
        found[f"{field} off by one"] = {**right, field: right[field] + 1}
    found["per-token and per-sequence swapped"] = {
        **right,
        "kv_bytes_per_token": right["kv_bytes_per_sequence"],
        "kv_bytes_per_sequence": right["kv_bytes_per_token"],
    }
    return {name: wrong for name, wrong in found.items() if wrong != right}


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_approves_the_right_plan_of_every_case(number: int, tmp_path: Path) -> None:
    case = cases.case(number)
    assert plan_with(case) == plain_output(plan(case))
    verdict = verify_task(number, plain_output(plan(case)), case, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0, verdict["message"]


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_refuses_every_wrong_plan(number: int, tmp_path: Path) -> None:
    case = cases.case(number)
    for name, output in mutants(case).items():
        verdict = verify_task(number, output, case, tmp_path)
        assert verdict["valid"] is False, f"case {number}: {name} was approved"
        assert 0.0 <= verdict["score"] < 1.0, f"case {number}: {name} scored {verdict['score']}"


def test_every_mistake_changes_the_answer_of_some_case() -> None:
    caught = set()
    for number in range(1, CASES + 1):
        caught |= set(mutants(cases.case(number)))
    assert set(MISTAKES) <= caught, f"never caught: {set(MISTAKES) - caught}"


def test_each_wrong_plan_says_what_is_wrong(tmp_path: Path) -> None:
    case = cases.case(2)
    right = plan_with(case)
    verdict = verify_task(2, plan_with(case, "GB read as GiB"), case, tmp_path)
    assert verdict["message"].startswith(f"usable_bytes is {plan_with(case, 'GB read as GiB')['usable_bytes']}")
    assert verdict["message"].endswith("1 more field(s) wrong: max_batch") and verdict["score"] == 4 / 6
    verdict = verify_task(2, {**right, "max_batch": right["max_batch"] + 1}, case, tmp_path)
    assert verdict["message"] == f"max_batch is {right['max_batch'] + 1}, expected {right['max_batch']}"
    assert verify_task(4, plan_with(cases.case(4), "negative max_batch"), cases.case(4), tmp_path)["message"].startswith(
        "max_batch is -"
    )


def test_refuses_without_the_reference_of_the_case_asked_for(tmp_path: Path) -> None:
    case = cases.case(1)
    right = plain_output(plan(case))
    assert verify_task(1, right, None, tmp_path)["valid"] is False
    assert verify_task(2, right, case, tmp_path)["valid"] is False
    broken = replace(case, config_json=case.config_json.replace('"num_key_value_heads": 8', '"num_key_value_heads": 5'))
    assert "cannot be planned" in verify_task(1, right, broken, tmp_path)["message"]


@pytest.mark.parametrize(
    "output",
    [
        {field: 1 for field in FIELDS[:-1]},
        {**{field: 1 for field in FIELDS}, "max_batch": "16"},
        {**{field: 1 for field in FIELDS}, "max_batch": 1.5},
        {**{field: 1 for field in FIELDS}, "max_batch": True},
        {**{field: 1 for field in FIELDS}, "max_batch": None},
        {**{field: 1 for field in FIELDS}, "fits": True},
    ],
)
def test_the_contract_refuses_a_malformed_output_before_the_verifier(output: dict) -> None:
    with pytest.raises(ValueError):
        build_output(output, PlanOutput)


def test_checks_what_the_types_cannot_say() -> None:
    assert verify_input(Tool.READ_CASE, {"case": 1})["valid"] is True
    assert verify_input(Tool.READ_CASE, {"case": 0})["valid"] is False
    assert verify_input(Tool.CASE_RECORD, {"case": CASES + 1})["valid"] is False
    assert verify_input(Tool.CALCULATE, {"expression": "2 * (3 + 4) // 5"})["valid"] is True
    assert verify_input(Tool.CALCULATE, {"expression": "  "})["valid"] is False
    assert verify_input(Tool.CALCULATE, {"expression": "hidden * 2"})["valid"] is False
    assert verify_input(Tool.CALCULATE, {"expression": "1.5 * 2"})["valid"] is False
    assert verify_input(Tool.CALCULATE, {"expression": "1" * 501})["valid"] is False


def test_checks_each_calculation() -> None:
    given = {"expression": " 80 * 2**30 - 7 // 2 "}
    assert verify_output(Tool.CALCULATE, given, {"expression": "80 * 2**30 - 7 // 2", "value": 85899345917})["valid"]
    assert not verify_output(Tool.CALCULATE, given, {"expression": "80 * 2**30 - 7 // 2", "value": 85899345916})["valid"]
    assert not verify_output(Tool.CALCULATE, given, {"expression": "80 * 2**30", "value": 85899345920})["valid"]


def test_checks_the_cases() -> None:
    for number in range(1, CASES + 1):
        case = plain_output(cases.case(number))
        assert verify_output(Tool.READ_CASE, {"case": number}, case)["valid"] is True
        assert verify_output(Tool.CASE_RECORD, {"case": number}, case)["valid"] is True
    case = plain_output(cases.case(1))
    assert verify_output(Tool.READ_CASE, {"case": 2}, case)["valid"] is False
    for broken in [
        {"gpu_memory": "80GiB"},
        {"reserved_memory": "4 TB"},
        {"kv_cache_dtype": "int4"},
        {"context_length": 0},
        {"config_json": "{"},
        {"config_json": case["config_json"].replace('"hidden_size": 4096', '"hidden_size": -4096')},
        {"config_json": case["config_json"].replace('"torch_dtype": "bfloat16"', '"torch_dtype": "int8"')},
        {"config_json": case["config_json"].replace('"attention_bias": false', '"attention_bias": "no"')},
    ]:
        assert verify_output(Tool.READ_CASE, {"case": 1}, {**case, **broken})["valid"] is False, broken
