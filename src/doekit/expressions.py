"""Evaluate the small maths expressions written in config files.

Constraints and simulated responses are written as text, for example
``gelma + alginate`` or ``50 + 3 * pressure - 0.2 * pressure ** 2``. This module
parses them and allows only arithmetic and a short list of functions, so a shared
config file can't run arbitrary code the way Python's ``eval`` would let it.
"""

from __future__ import annotations

import ast
import operator

import numpy as np

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "where": np.where,
}


def _parse(expression: str) -> ast.Expression:
    try:
        return ast.parse(str(expression), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Could not read the expression '{expression}'.") from error


def names_in(expression: str) -> set[str]:
    """Return the variable names an expression uses, leaving out function names."""
    tree = _parse(expression)
    functions = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return names - functions


def evaluate(expression: str, variables: dict):
    """Evaluate an expression, with each name taken from ``variables``."""

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(
                    f"'{node.id}' in the expression '{expression}' is not a known factor."
                )
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _COMPARE
        ):
            return _COMPARE[type(node.ops[0])](walk(node.left), walk(node.comparators[0]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*[walk(argument) for argument in node.args])
        raise ValueError(
            f"The expression '{expression}' uses something that isn't allowed. "
            f"Use + - * / **, comparisons, and these functions: {', '.join(sorted(_FUNCTIONS))}."
        )

    return walk(_parse(expression))
