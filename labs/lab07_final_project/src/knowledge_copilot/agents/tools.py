"""Bounded function tools with AST-safe calculation and retrieval."""
import ast
import json
import math
import operator
from typing import Any, Callable, Dict, List, Optional

from ..domain.enums import ToolCallOrigin
from ..domain.models import ToolExecutionRecord


def safe_calculate(expression: str) -> Dict[str, Any]:
    """Evaluate strictly bounded numeric arithmetic.
    
    Permitted: numeric literals, parentheses, +, -, *, /
    Forbidden: function calls, imports, variables, attributes, exponentiation.
    """
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must be a non-empty string with at most 200 characters.")
    
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression exceeds allowed AST complexity limit (64 nodes).")
    
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            val = evaluate(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left_val = evaluate(node.left)
            right_val = evaluate(node.right)
            if isinstance(node.op, ast.Div) and right_val == 0:
                raise ZeroDivisionError("Division by zero in arithmetic expression.")
            return operations[type(node.op)](left_val, right_val)
        else:
            raise ValueError("Only numeric literals, parentheses and +, -, *, / are permitted.")

    result = evaluate(tree.body)
    if not math.isfinite(result) or abs(result) > 1e15:
        raise ValueError("Calculated numeric result is out of supported finite bounds (<= 1e15).")
    
    return {"expression": expression, "result": result}
