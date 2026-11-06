"""The plan-kv-cache task's check: every field of the plan equals the verifier's own plan of the case."""

from .checks import result
from .plan import FIELDS, case_problem, expected_plan


def verify_task(arguments, output, reference, workspace):
    """``reference``: the contract CaseOutput of the case asked for, pinned before the agent ran."""
    if reference is None or reference.case != arguments.case:
        return result(False, f"the pinned reference is not case {arguments.case}")
    problem = case_problem(reference)
    if problem:
        return result(False, f"the pinned case cannot be planned: {problem}")
    expected = expected_plan(reference)
    submitted = {
        "param_count": output.param_count,
        "weight_bytes": output.weight_bytes,
        "kv_bytes_per_token": output.kv_bytes_per_token,
        "kv_bytes_per_sequence": output.kv_bytes_per_sequence,
        "usable_bytes": output.usable_bytes,
        "max_batch": output.max_batch,
    }
    wrong = [field for field in FIELDS if submitted[field] != expected[field]]
    score = (len(FIELDS) - len(wrong)) / len(FIELDS)
    if wrong:
        first = wrong[0]
        message = f"{first} is {submitted[first]}, expected {expected[first]}"
        if len(wrong) > 1:
            message += f"; {len(wrong) - 1} more field(s) wrong: {', '.join(wrong[1:])}"
        return result(False, message, score=score, expected=expected, wrong=wrong)
    return result(True, "every field of the plan is exact", max_batch=expected["max_batch"])
