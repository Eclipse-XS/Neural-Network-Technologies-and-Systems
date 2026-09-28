"""Read-only allowlisted reports and bounded AST arithmetic; no shell or web."""
import ast
import json
import math
import operator
from pathlib import Path, PureWindowsPath
from agents import function_tool
from .config import WORKSPACE


class WorkspaceTools:
    def __init__(self, workspace=WORKSPACE):
        self.workspace = Path(workspace).resolve()
        self.manifest = json.loads((self.workspace / "manifest.json").read_text(encoding="utf-8"))
        self.allowed = {item["workspace_filename"] for item in self.manifest}

    def path(self, filename):
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute() or PureWindowsPath(filename).is_absolute() or any(x in filename for x in ("..", "/", "\\", ":")) or filename not in self.allowed:
            raise ValueError("File is not an allowed workspace report")
        path = (self.workspace / filename).resolve()
        if not path.is_relative_to(self.workspace) or not path.is_file():
            raise ValueError("Report is missing or resolves outside workspace")
        return path

    def list_files(self):
        return [{"filename": n, "size_bytes": self.path(n).stat().st_size} for n in sorted(self.allowed)]

    def search(self, query: str, filename: str | None = None):
        if not query.strip() or len(query) > 200:
            raise ValueError("Query must contain 1–200 characters")
        names = [filename] if filename is not None else sorted(self.allowed)
        results = []
        for name in names:
            for number, line in enumerate(self.path(name).read_text(encoding="utf-8").splitlines(), 1):
                if query.casefold() in line.casefold():
                    results.append({"filename": name, "line": number, "text": line[:650]})
                    if len(results) == 8:
                        return {"matches": results, "limit_reached": True}
        return {"matches": results, "limit_reached": False}

    def read(self, filename: str, start_line: int = 1, end_line: int | None = None):
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            raise ValueError("start_line must be a positive integer")
        end_line = start_line + 59 if end_line is None else end_line
        if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < start_line or end_line - start_line >= 80:
            raise ValueError("Read 1–80 lines in ascending order")
        lines = self.path(filename).read_text(encoding="utf-8").splitlines()
        selected = []
        size = 0
        for n in range(start_line, min(end_line, len(lines)) + 1):
            value = lines[n - 1][:1000]
            if size + len(value) > 6500:
                break
            selected.append({"line": n, "text": value}); size += len(value)
        return {"filename": filename, "total_lines": len(lines), "lines": selected,
                "next_line": selected[-1]["line"] + 1 if selected and selected[-1]["line"] < len(lines) else None}


def safe_calculate(expression: str):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must contain 1–200 characters")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            result = operations[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("Only numbers, parentheses and + - * / are allowed")
        if not math.isfinite(result) or abs(result) > 1e15:
            raise ValueError("Arithmetic magnitude exceeds safe limit")
        return result
    return {"expression": expression, "result": visit(tree.body)}


def tool_error(context, error):
    """A concise recoverable tool error; the hook preserves it in local logs."""
    return json.dumps({"error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False)


def build_tools(workspace=WORKSPACE):
    store = WorkspaceTools(workspace)
    @function_tool(failure_error_function=tool_error)
    def list_workspace_files() -> dict:
        """List available coursework reports and byte sizes. Use to discover filenames."""
        return {"files": store.list_files()}

    @function_tool(failure_error_function=tool_error)
    def search_workspace(query: str, filename: str | None = None) -> dict:
        """Search reports for a case-insensitive literal phrase. Use for local facts.

        Args:
            query: Literal phrase; Ukrainian report words may work better than English.
            filename: Optional exact report filename to restrict search.
        """
        return store.search(query, filename)

    @function_tool(failure_error_function=tool_error)
    def read_workspace_file(filename: str, start_line: int = 1, end_line: int | None = None) -> dict:
        """Read numbered report lines to obtain exact evidence. Defaults to 60 lines.

        Args:
            filename: Exact filename from the workspace listing.
            start_line: First line, numbered from one.
            end_line: Last inclusive line; at most 80 lines per call.
        """
        return store.read(filename, start_line, end_line)

    @function_tool(failure_error_function=tool_error)
    def calculate(expression: str) -> dict:
        """Calculate derived arithmetic from retrieved values, never execute Python code.

        Args:
            expression: Numeric expression using parentheses and + - * / only.
        """
        return safe_calculate(expression)
    return [list_workspace_files, search_workspace, read_workspace_file, calculate]
