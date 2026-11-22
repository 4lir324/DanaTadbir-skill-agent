"""Safe math evaluator: walks the AST and allows only whitelisted nodes.

No `eval`/`exec`. Anything outside the whitelist is rejected, and a few guards
stop expressions that would hang the process (huge powers, huge factorials).
"""

import ast
import math
import operator
from collections.abc import Callable

from skill_agent.text import normalize

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_FACTORIAL = 170
MAX_RESULT_BITS = 100_000  # blocks nested powers such as ((9**999)**999)**999

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

# Characters people (and LLMs) commonly write instead of ASCII operators.
_SYMBOL_MAP = str.maketrans({"^": "**", "×": "*", "÷": "/", "−": "-", "٫": ".", "،": ","})


class CalculatorError(ValueError):
    """The expression is invalid, unsafe, or cannot be computed."""


def clean_expression(expression: str) -> str:
    """Normalize digits (Persian/Arabic), operator symbols and whitespace."""
    return normalize(expression).translate(_SYMBOL_MAP).strip()


def evaluate(expression: str) -> float:
    """Evaluate a math expression safely; raise CalculatorError on any problem."""
    cleaned = clean_expression(expression)
    if not cleaned:
        raise CalculatorError("Empty expression")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression is too long")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as error:
        raise CalculatorError(f"Invalid expression: {cleaned}") from error
    try:
        return _eval_node(tree.body)
    except CalculatorError:
        raise
    except ZeroDivisionError as error:
        raise CalculatorError("Division by zero") from error
    except (OverflowError, ValueError) as error:
        raise CalculatorError(f"Cannot compute: {error}") from error


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power_size(left, right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        return _call_function(node)
    raise CalculatorError("Unsupported expression")


def _check_power_size(base: float, exponent: float) -> None:
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError("Exponent is too large")
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if base.bit_length() * exponent > MAX_RESULT_BITS:
            raise CalculatorError("Result is too large")


def _call_function(node: ast.Call) -> float:
    name = node.func.id
    if name not in _FUNCTIONS:
        raise CalculatorError(f"Unsupported function: {name}")
    args = [_eval_node(arg) for arg in node.args]
    if name == "factorial" and (len(args) != 1 or not 0 <= args[0] <= MAX_FACTORIAL):
        raise CalculatorError("Factorial argument out of range")
    try:
        return _FUNCTIONS[name](*args)
    except TypeError as error:
        raise CalculatorError(f"Bad arguments for {name}") from error


def format_number(value: float) -> str:
    """Human-friendly number: integers without '.0', floats to 12 significant digits."""
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".12g")
