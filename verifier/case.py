"""read_case's and case_record's checks: the case asked for, one the rules can plan."""

from .checks import case_number_problem, result
from .plan import case_problem


def verify_input(value):
    problem = case_number_problem(value.case)
    if problem:
        return result(False, problem)
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.case != input_value.case:
        return result(False, f"asked for case {input_value.case}, got case {output_value.case}")
    problem = case_problem(output_value)
    if problem:
        return result(False, problem)
    return result(True, "the case can be planned")
