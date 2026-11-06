from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from slimconfig import plain_output

import verifier.calculate
import verifier.plan
from contract import CASES, CalculateInput, CaseInput
from plugin import calculate, case_record, cases, read_case
from plugin.cases import CaseSpec, build_case
from plugin.plan import plan


def value(expression: str, workspace: Path) -> int:
    return calculate(CalculateInput(expression=expression), workspace).value


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("80 * 2**30 - 4 * 2**30", 76 * 2**30),
        ("-(3 + 4) // 2", -4),
        ("2 ** 3 ** 2", 512),
        ("-2 ** 2", -4),
        ("--3", 3),
        ("+7 // 2", 3),
        ("  6 * 7\t", 42),
        ("0", 0),
        ("2 ** 64", 2**64),
    ],
)
def test_calculate_is_python_integer_arithmetic(expression: str, expected: int, tmp_path: Path) -> None:
    assert value(expression, tmp_path) == expected
    assert verifier.calculate.value_of(expression.strip()) == expected


@pytest.mark.parametrize(
    "expression",
    ["x + 1", "abs(-1)", "1.5 * 2", "1 / 2", "7 % 3", "2 ** 65", "2 ** -1", "10 ** 29 * 100", "", "1 +", "(1", "1 // 0",
     "01", "1e3", "2 ^ 3", "__import__('os')", "1" * 501],
)
def test_calculate_refuses_anything_else(expression: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        value(expression, tmp_path)


def random_expression(rng: random.Random, depth: int = 0) -> str:
    if depth > 3 or rng.random() < 0.3:
        return str(rng.choice([0, 1, 2, 3, 7, 10, 64, 128, 1024, 4096, 128256]))
    kind = rng.choice(["+", "-", "*", "//", "**", "neg", "paren"])
    if kind == "neg":
        return "-" + random_expression(rng, depth + 1)
    if kind == "paren":
        return "(" + random_expression(rng, depth + 1) + ")"
    if kind == "**":
        return random_expression(rng, depth + 1) + " ** " + str(rng.randint(0, 4))
    return random_expression(rng, depth + 1) + f" {kind} " + random_expression(rng, depth + 1)


@pytest.mark.parametrize("seed", range(300))
def test_the_verifier_and_the_plugin_calculate_alike(seed: int, tmp_path: Path) -> None:
    expression = random_expression(random.Random(seed))
    try:
        expected = eval(expression, {"__builtins__": {}})  # the test's oracle: Python itself
    except ZeroDivisionError:
        expected = None
    try:
        got = value(expression, tmp_path)
    except ValueError as error:
        # calculate refuses only division by zero and steps past its caps (exponent 64, magnitude 10**30).
        assert expected is None or "exponent" in str(error) or "10**30" in str(error)
        return
    assert got == expected
    assert verifier.calculate.value_of(expression) == expected


def random_spec(rng: random.Random) -> CaseSpec:
    heads = rng.choice([8, 12, 16, 32, 40, 64])
    hidden = heads * rng.choice([64, 80, 96, 128])
    config = {
        "hidden_size": hidden,
        "intermediate_size": rng.randrange(256, 8 * hidden, 256),
        "num_hidden_layers": rng.randint(1, 96),
        "num_attention_heads": heads,
        "vocab_size": rng.randint(1000, 260000),
        "max_position_embeddings": rng.choice([2048, 4096, 32768]),
        "torch_dtype": rng.choice(["float32", "bfloat16", "float16"]),
    }
    if rng.random() < 0.7:
        config["num_key_value_heads"] = rng.choice([k for k in (1, 2, 4, 8, heads) if heads % k == 0])
    if rng.random() < 0.3:
        config["head_dim"] = rng.choice([64, 96, 128, 256])
    for key in ("attention_bias", "tie_word_embeddings"):
        if rng.random() < 0.6:
            config[key] = rng.random() < 0.5
    return CaseSpec(
        config=config,
        gpu_memory=f"{rng.choice([16, 24, 40, 48, 80, 141])} {rng.choice(['GiB', 'GB'])}",
        reserved_memory=f"{rng.randint(0, 4096)} {rng.choice(['MiB', 'MB'])}",
        context_length=rng.choice([512, 2048, 4096, 8192, 32768, 131072]),
        kv_cache_dtype=rng.choice(["auto", "fp8", "bfloat16", "float16"]),
        tests="random",
    )


@pytest.mark.parametrize("seed", range(500))
def test_the_verifier_and_the_plugin_plan_alike(seed: int) -> None:
    case = build_case(0, random_spec(random.Random(seed)))
    assert verifier.plan.case_problem(case) is None
    assert verifier.plan.expected_plan(case) == plain_output(plan(case))


def test_cases_are_the_same_on_every_run(tmp_path: Path) -> None:
    for number in range(1, CASES + 1):
        assert read_case(CaseInput(case=number), tmp_path) == case_record(CaseInput(case=number), tmp_path)
        assert cases.case(number) == cases.case(number)
    with pytest.raises(ValueError):
        cases.case(CASES + 1)


def test_each_case_tests_what_it_says() -> None:
    def config(number: int) -> dict:
        return json.loads(cases.case(number).config_json)

    assert config(1)["num_key_value_heads"] < config(1)["num_attention_heads"]
    assert "num_key_value_heads" not in config(2) and cases.case(2).gpu_memory.endswith(" GB")
    assert config(3)["num_key_value_heads"] == 1 and config(3)["tie_word_embeddings"] is True
    assert cases.case(3).kv_cache_dtype == "fp8"
    four = plan(cases.case(4))
    assert four.usable_bytes < 0 and four.max_batch == 0
    assert config(5)["head_dim"] != config(5)["hidden_size"] // config(5)["num_attention_heads"]
    assert config(5)["attention_bias"] is True and config(5)["torch_dtype"] == "float32"
    six = plan(cases.case(6))
    assert six.usable_bytes == 16 * six.kv_bytes_per_sequence and six.max_batch == 16
    for number in range(1, CASES + 1):
        assert config(number)["max_position_embeddings"] != cases.case(number).context_length
    # Published parameter counts: Llama-3-8B, Llama-2-7B, Gemma-2B and Llama-3-70B.
    assert [plan(cases.case(n)).param_count for n in (1, 2, 3, 4)] == [
        8030261248,
        6738415616,
        2506172416,
        70553706496,
    ]
