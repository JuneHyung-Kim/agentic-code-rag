from indexing.storage.graph_store import get_graph_store

# Tool list:
# 1. search_codebase - 시맨틱 + 키워드 하이브리드 검색 (구현됨, SearchTool)
# 2. read_file - 특정 파일 읽기 (구현됨, FileSystemTools)
# 3. list_directory - 디렉토리 구조 탐색 (구현됨, FileSystemTools)
# 4. get_callers - 특정 함수를 호출하는 코드 찾기 (구현됨, RelatedCodeTool)
# 5. get_callees - 특정 함수가 호출하는 코드 찾기 (구현됨, RelatedCodeTool)
# 6. get_symbol_definition - 심볼 정의 위치 찾기 (TODO)
# 7. get_symbol_references - 심볼 참조 위치 찾기 (TODO)
# 8. search_by_pattern - 정규식 패턴 검색 (TODO)


class RelatedCodeTool:
    """
    Tools for traversing the code property graph (GraphStore).
    Provides caller/callee graph traversal capabilities.
    """

    def __init__(self):
        self.graph_store = get_graph_store()

    def get_callers(self, function_name: str) -> str:
        """
        Find functions that call the given function.
        Returns formatted string with caller information.
        """
        callers = self.graph_store.get_callers(function_name)

        if not callers:
            return f"No callers found for '{function_name}'. The function may not be indexed or has no callers in the graph."

        lines = [f"Functions that call '{function_name}':"]
        for caller in callers:
            file_path = caller.get("file_path", "unknown")
            name = caller.get("name", caller.get("node_id", "unknown"))
            node_type = caller.get("type", "unknown")
            lines.append(f"  - {name} ({node_type}) in {file_path}")

        return "\n".join(lines)

    def get_callees(self, function_name: str) -> str:
        """
        Find functions that the given function calls.
        Returns formatted string with callee information.
        """
        callees = self.graph_store.get_callees(function_name)

        if not callees:
            return f"No callees found for '{function_name}'. The function may not be indexed or makes no calls in the graph."

        lines = [f"Functions called by '{function_name}':"]
        for callee in callees:
            file_path = callee.get("file_path", "unknown")
            name = callee.get("name", callee.get("node_id", "unknown"))
            node_type = callee.get("type", "unknown")
            lines.append(f"  - {name} ({node_type}) in {file_path}")

        return "\n".join(lines)

    def get_call_chain(
        self, function_name: str, direction: str = "callees", max_depth: int = 3
    ) -> str:
        """Trace an N-hop call chain and return a tree-formatted string."""
        chain = self.graph_store.get_call_chain(
            function_name, direction=direction, max_depth=max_depth
        )

        if not chain:
            return (
                f"No call chain found for '{function_name}' "
                f"(direction={direction}). The function may not be indexed."
            )

        lines = [
            f"Call chain ({direction}) for '{function_name}' (max depth {max_depth}):"
        ]
        for entry in chain:
            indent = "  " * entry["depth"]
            name = entry["name"]
            sym_type = entry["type"]
            file_path = entry.get("file_path", "unknown")
            lines.append(f"{indent}{name} ({sym_type}) in {file_path}")

        return "\n".join(lines)

    def get_related(self, symbol_name: str) -> str:
        """
        Find related code (callers, callees) for a given symbol name.
        Combines results from get_callers and get_callees.
        """
        callers_result = self.get_callers(symbol_name)
        callees_result = self.get_callees(symbol_name)

        return f"{callers_result}\n\n{callees_result}"
