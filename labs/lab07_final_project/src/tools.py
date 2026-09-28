"""Three bounded function tools; evidence retrieval performs no generation."""
import ast
import json
import math
import operator
from agents import function_tool


def safe_calculate(expression: str) -> dict:
    """Evaluate bounded arithmetic only; names, calls and attributes are rejected."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError('Expression must contain 1–200 characters')
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('Expression too complex')
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            result = operations[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError('Only numeric literals, parentheses and + - * / are permitted')
        if not math.isfinite(result) or abs(result) > 1e15:
            raise ValueError('Numeric value out of bounds')
        return result
    return dict(expression=expression, result=visit(tree.body))


def tool_error(context, error):
    return json.dumps(dict(error_type=type(error).__name__, error=str(error)), ensure_ascii=False)


def build_tools(kb):
    @function_tool(failure_error_function=tool_error)
    def search_knowledge(query: str, k: int = 5, source_id: str | None = None) -> dict:
        """Retrieve evidence chunks from the local report vector index. No answer generation.

        Args:
            query: Focused semantic query. Reports use Ukrainian text and English model IDs.
            k: Number of neighboring chunks, 1–8; normally 5.
            source_id: Optional exact lab03, lab04, lab05 or lab06 to restrict to that report.
        """
        chunks = kb.search(query, k, source_id)
        return dict(chunks=[c.to_dict() for c in chunks],
            evidence_note='Top-k candidates are not proof of relevance. Use only text that supports the claim; otherwise state insufficient evidence.',
            score_semantics='cosine distance; lower is closer', empty=not chunks)

    @function_tool(failure_error_function=tool_error)
    def list_knowledge_sources() -> dict:
        """List indexed lab numbers, IDs, titles and filenames, without report contents."""
        return dict(sources=kb.sources())

    @function_tool(failure_error_function=tool_error)
    def calculate(expression: str) -> dict:
        """Compute exact derived arithmetic using retrieved operands.

        Args:
            expression: Numeric expression with parentheses and + - * / only.
        """
        return safe_calculate(expression)
    return [search_knowledge, list_knowledge_sources, calculate]
