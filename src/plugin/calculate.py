"""calculate: exact integer arithmetic, so the agent never multiplies large numbers in its head."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from contract import MAX_EXPRESSION, CalculateInput, CalculateOutput

ALLOWED_CHARACTERS = re.compile(r"[0-9+\-*/() \t]*")
# The largest exponent, and the largest magnitude any step may reach: far beyond any memory size.
MAX_EXPONENT = 64
MAX_MAGNITUDE = 10**30
OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Pow: lambda a, b: a**b,
}


def evaluate(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = evaluate(node.operand)
        value = -operand if isinstance(node.op, ast.USub) else operand
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            raise ValueError("division by zero")
        if isinstance(node.op, ast.Pow) and not 0 <= right <= MAX_EXPONENT:
            raise ValueError(f"an exponent must be 0 to {MAX_EXPONENT}, not {right}")
        value = OPERATORS[type(node.op)](left, right)
    else:
        raise ValueError("only integers, + - * // ** and parentheses are allowed")
    if abs(value) > MAX_MAGNITUDE:
        raise ValueError("the value grows past 10**30")
    return value


def calculate(given: CalculateInput, workspace: Path) -> CalculateOutput:
    expression = given.expression.strip()
    if not expression or len(expression) > MAX_EXPRESSION:
        raise ValueError(f"the expression must be 1 to {MAX_EXPRESSION} characters")
    if not ALLOWED_CHARACTERS.fullmatch(expression):
        raise ValueError("only integers, + - * // ** and parentheses are allowed")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"not an expression: {expression!r}") from None
    return CalculateOutput(expression=expression, value=evaluate(tree.body))
