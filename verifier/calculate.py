"""calculate's checks, with the verifier's own parser: Python's integer arithmetic on + - * // ** and parentheses."""

import re

from contract import MAX_EXPRESSION

from .checks import result

CHARACTERS = re.compile(r"[0-9+\-*/() \t]*")
TOKEN = re.compile(r"[ \t]*(?:(\d+)|(\*\*|//|[-+*()]))")
MAX_EXPONENT = 64


def tokens_of(text):
    tokens = []
    position = 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"unexpected {text[position:]!r}")
        tokens.append(int(match.group(1)) if match.group(1) is not None else match.group(2))
        position = match.end()
    return tokens


# Each parse_* takes the tokens and a position, and returns a value and the position after it,
# following Python's grammar: sum := product (+|- product)*; product := unary (*|// unary)*;
# unary := (+|-) unary | power; power := atom [** unary]; atom := integer | ( sum ).
def parse_sum(tokens, at):
    value, at = parse_product(tokens, at)
    while at < len(tokens) and tokens[at] in ("+", "-"):
        operator = tokens[at]
        right, at = parse_product(tokens, at + 1)
        value = value + right if operator == "+" else value - right
    return value, at


def parse_product(tokens, at):
    value, at = parse_unary(tokens, at)
    while at < len(tokens) and tokens[at] in ("*", "//"):
        operator = tokens[at]
        right, at = parse_unary(tokens, at + 1)
        if operator == "//" and right == 0:
            raise ValueError("division by zero")
        value = value * right if operator == "*" else value // right
    return value, at


def parse_unary(tokens, at):
    if at < len(tokens) and tokens[at] in ("+", "-"):
        value, after = parse_unary(tokens, at + 1)
        return (-value if tokens[at] == "-" else value), after
    return parse_power(tokens, at)


def parse_power(tokens, at):
    base, at = parse_atom(tokens, at)
    if at < len(tokens) and tokens[at] == "**":
        exponent, at = parse_unary(tokens, at + 1)
        if not 0 <= exponent <= MAX_EXPONENT:
            raise ValueError(f"exponent {exponent} is out of range")
        return base**exponent, at
    return base, at


def parse_atom(tokens, at):
    if at >= len(tokens):
        raise ValueError("the expression ends too soon")
    token = tokens[at]
    if type(token) is int:
        return token, at + 1
    if token == "(":
        value, at = parse_sum(tokens, at + 1)
        if at >= len(tokens) or tokens[at] != ")":
            raise ValueError("a parenthesis is not closed")
        return value, at + 1
    raise ValueError(f"unexpected {token!r}")


def value_of(text):
    """The expression's value, or raises ValueError."""
    tokens = tokens_of(text)
    value, at = parse_sum(tokens, 0)
    if at != len(tokens):
        raise ValueError(f"unexpected {tokens[at]!r}")
    return value


def verify_input(value):
    expression = value.expression.strip()
    if not 1 <= len(expression) <= MAX_EXPRESSION:
        return result(False, f"the expression must be 1 to {MAX_EXPRESSION} characters")
    if not CHARACTERS.fullmatch(expression):
        return result(False, "only integers, + - * // ** and parentheses are allowed")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.expression != input_value.expression.strip():
        return result(False, "the output's expression is not the one given")
    try:
        expected = value_of(output_value.expression)
    except ValueError as error:
        return result(False, f"the expression cannot be evaluated: {error}")
    if output_value.value != expected:
        return result(False, f"{output_value.expression} is {expected}, not {output_value.value}")
    return result(True, "the value is exact")
