from indexing.storage.graph_store import get_graph_store

# TODO: 확장 예정 tool list
# 1. search_codebase - 시맨틱 + 키워드 하이브리드 검색 (구현됨, SearchTool)
# 2. read_file - 특정 파일 읽기 (구현됨, FileSystemTools)
# 3. list_directory - 디렉토리 구조 탐색 (구현됨, FileSystemTools)
# 4. get_callers - 특정 함수를 호출하는 코드 찾기 (GraphStore 활용 필요)
# 5. get_callees - 특정 함수가 호출하는 코드 찾기 (GraphStore 활용 필요)
# 6. get_symbol_definition - 심볼 정의 위치 찾기
# 7. get_symbol_references - 심볼 참조 위치 찾기
# 8. search_by_pattern - 정규식 패턴 검색


class RelatedCodeTool:
    """
    Tools for traversing the code property graph (GraphStore).

    TODO: Implement full GraphStore integration for:
    - Caller/callee graph traversal
    - Symbol definition lookup
    - Symbol reference lookup
    """

    def __init__(self):
        self.graph_store = get_graph_store()

    def get_callers(self, function_name: str) -> str:
        """
        Find functions that call the given function.

        TODO: Implement using GraphStore.get_callers() once available.
        Currently returns placeholder.
        """
        # TODO: Implement graph traversal
        # callers = self.graph_store.get_callers(function_name)
        # return formatted result
        return f"[get_callers not yet implemented] Searching for callers of '{function_name}'..."

    def get_callees(self, function_name: str) -> str:
        """
        Find functions that the given function calls.

        TODO: Implement using GraphStore.get_callees() once available.
        Currently returns placeholder.
        """
        # TODO: Implement graph traversal
        # callees = self.graph_store.get_callees(function_name)
        # return formatted result
        return f"[get_callees not yet implemented] Searching for callees of '{function_name}'..."

    def get_related(self, symbol_name: str) -> str:
        """
        Find related code (callers, callees) for a given symbol name.

        TODO: Combine get_callers and get_callees results.
        Currently returns placeholder.
        """
        return f"[get_related not yet implemented] Searching for relations of '{symbol_name}'..."
