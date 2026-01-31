from tree_sitter import Language, Parser, Query, QueryCursor, Node
from typing import List, Optional, Set
from ..schema import CodeNode

class PythonParser:
    def __init__(self, parser: Parser, language: Language):
        self.parser = parser
        self.language = language
        self._node_types = self._load_node_types()
        self._definition_query = self._build_definition_query()
        self._import_query = self._build_import_query()
        self._call_query = self._build_call_query()

    def _load_node_types(self) -> Set[str]:
        node_types = set()
        count = self.language.node_kind_count
        if not isinstance(count, int):
            count = self.language.node_kind_count()
        for i in range(count):
            node_types.add(self.language.node_kind_for_id(i))
        return node_types

    def _has_node_type(self, node_type: str) -> bool:
        return node_type in self._node_types

    def _build_definition_query(self) -> Optional[Query]:
        parts = []
        if self._has_node_type("function_definition"):
            parts.append("(function_definition) @function")
        if self._has_node_type("async_function_definition"):
            parts.append("(async_function_definition) @function")
        if self._has_node_type("class_definition"):
            parts.append("(class_definition) @class")
        if self._has_node_type("decorated_definition"):
            if self._has_node_type("function_definition"):
                parts.append("(decorated_definition (function_definition) @function)")
            if self._has_node_type("async_function_definition"):
                parts.append("(decorated_definition (async_function_definition) @function)")
            if self._has_node_type("class_definition"):
                parts.append("(decorated_definition (class_definition) @class)")
        if self._has_node_type("assignment"):
            parts.append("(assignment) @assignment")
        if not parts:
            return None
        return Query(self.language, "\n".join(parts))

    def _build_import_query(self) -> Optional[Query]:
        parts = []
        if self._has_node_type("import_statement"):
            parts.append("(import_statement) @import")
        if self._has_node_type("import_from_statement"):
            parts.append("(import_from_statement) @import")
        if not parts:
            return None
        return Query(self.language, "\n".join(parts))

    def parse(self, code: str, file_path: str) -> List[CodeNode]:
        tree = self.parser.parse(bytes(code, "utf8"))
        nodes = []
        
        # 1. Extract File-level Imports
        file_imports = self._extract_imports(tree.root_node, code)
        
        # 2. Extract Definitions
        if not self._definition_query:
            return nodes
        cursor = QueryCursor(self._definition_query)
        captures = cursor.captures(tree.root_node)
        
        capture_list = []
        for node, name in self._iter_captures(captures):
            capture_list.append((node, name))
        
        # Sort by position to handle hierarchy
        capture_list.sort(key=lambda x: x[0].start_byte)
        
        processed_ids = set()
        
        for node, capture_type in capture_list:
            if node.id in processed_ids:
                continue
            
            # Only process supported capture types
            if capture_type not in ['function', 'class', 'assignment']:
                continue
                
            processed_ids.add(node.id)
            
            # Skip non-module assignments (avoid locals)
            if capture_type == 'assignment':
                if not node.parent or node.parent.type != 'module':
                    continue

            # Extract basic info
            name = self._get_node_name(node, capture_type, code) or "anonymous"
            docstring = self._extract_docstring(node, code) if capture_type in ['function', 'class'] else None
            signature = self._extract_signature(node, capture_type, code) if capture_type == 'function' else None
            return_type = self._extract_return_type(node, code) if capture_type == 'function' else None
            
            # Extract arguments
            args = self._extract_arguments(node, code) if capture_type == 'function' else []
            calls = self._extract_function_calls(node, code) if capture_type in ['function', 'method'] else []

            parent_name = self._find_parent_class(node, code) if capture_type == 'function' else None
            if capture_type == 'assignment':
                node_type = "global_var"
            else:
                node_type = "method" if parent_name else capture_type
            
            # Create Node
            code_node = CodeNode(
                type=node_type if capture_type == 'function' else node_type,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                content=self._get_text(node, code),
                language='python',
                docstring=docstring,
                arguments=args,
                return_type=return_type,
                signature=signature,
                parent_name=parent_name,
                imports=file_imports,  # Share file-level imports,
                function_calls=calls
            )
            nodes.append(code_node)
            
        return nodes

    def _iter_captures(self, captures):
        if isinstance(captures, dict):
            for name, nodes in captures.items():
                for node in nodes:
                    yield node, name
        else:
            for node, name in captures:
                yield node, name

    def _extract_imports(self, root: Node, code: str) -> List[str]:
        imports = []
        if not self._import_query:
            return imports
        cursor = QueryCursor(self._import_query)
        captures = cursor.captures(root)
        for node, _ in self._iter_captures(captures):
            imports.append(self._get_text(node, code))
        return imports

    def _extract_docstring(self, node: Node, code: str) -> Optional[str]:
        body = node.child_by_field_name('body')
        if not body:
            return None
            
        for child in body.children:
            if child.type == 'expression_statement':
                # Check if it's a string literal (docstring)
                first = child.children[0]
                if first.type == 'string':
                    text = self._get_text(first, code)
                    return text.strip('"""').strip("'''").strip()
            elif child.type == 'comment':
                continue
            else:
                # Docstring must be the first statement
                break
        return None

    def _extract_return_type(self, node: Node, code: str) -> Optional[str]:
        return_type = node.child_by_field_name('return_type')
        if return_type:
            return self._get_text(return_type, code)
        return None

    def _extract_arguments(self, node: Node, code: str) -> List[str]:
        params = node.child_by_field_name('parameters')
        if not params:
            return []
        
        args = []
        for child in params.children:
            if child.type == 'identifier' or child.type.endswith('parameter'):
                args.append(self._get_text(child, code))
            elif child.type in ['list_splat', 'dictionary_splat', 'list_splat_pattern', 'dictionary_splat_pattern']:
                args.append(self._get_text(child, code))
        return args

    def _extract_signature(self, node: Node, capture_type: str, code: str) -> Optional[str]:
        if capture_type != 'function':
            return None
        body = node.child_by_field_name('body')
        if not body:
            return None
        signature = code[node.start_byte:body.start_byte].strip()
        return signature.rstrip(':').strip()

    def _get_node_name(self, node: Node, capture_type: str, code: str) -> str:
        if capture_type in ['function', 'class']:
            return self._get_text(node.child_by_field_name('name'), code)
        if capture_type == 'assignment':
            left = node.child_by_field_name('left')
            if left:
                found = self._first_identifier_in(left, code)
                if found:
                    return found
        return ""

    def _first_identifier_in(self, node: Node, code: str) -> str:
        if node.type == 'identifier':
            return self._get_text(node, code)
        for child in node.children:
            found = self._first_identifier_in(child, code)
            if found:
                return found
        return ""

    def _find_parent_class(self, node: Node, code: str) -> Optional[str]:
        parent = node.parent
        while parent:
            if parent.type == 'class_definition':
                name_node = parent.child_by_field_name('name')
                return self._get_text(name_node, code) if name_node else None
            parent = parent.parent
        return None

    def _get_text(self, node: Node, code: str) -> str:
        if not node:
            return ""
        return bytes(code, "utf8")[node.start_byte:node.end_byte].decode("utf8")

    def _build_call_query(self) -> Optional[Query]:
        if not self._has_node_type("call"):
            return None
        return Query(self.language, "(call function: (_) @call)")

    def _extract_function_calls(self, node: Node, code: str) -> List[str]:
        calls = set()
        if not self._call_query:
            return []
        
        cursor = QueryCursor(self._call_query)
        captures = cursor.captures(node)
        
        for captured_node, _ in self._iter_captures(captures):
            text = self._get_text(captured_node, code)
            if text:
                calls.add(text)
        
        return list(calls)
