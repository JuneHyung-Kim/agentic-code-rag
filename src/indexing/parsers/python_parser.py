from tree_sitter import Language, Parser, Query, QueryCursor, Node
from typing import List, Optional, Set
from ..schema import CodeNode

class PythonParser:
    def __init__(self, parser: Parser, language: Language):
        self.parser = parser
        self.language = language

    def parse(self, code: str, file_path: str) -> List[CodeNode]:
        tree = self.parser.parse(bytes(code, "utf8"))
        nodes = []
        
        # 1. Extract File-level Imports
        file_imports = self._extract_imports(tree.root_node, code)
        
        # 2. Extract Definitions
        query_scm = """
        (function_definition) @function
        (class_definition) @class
        (assignment) @assignment
        """
        
        query = Query(self.language, query_scm)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        
        capture_list = []
        for name, captured_nodes in captures.items():
            for node in captured_nodes:
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
                imports=file_imports  # Share file-level imports
            )
            nodes.append(code_node)
            
        return nodes

    def _extract_imports(self, root: Node, code: str) -> List[str]:
        imports = []
        # Simple traversal for imports
        # Python imports can be 'import_statement' or 'import_from_statement'
        query = Query(self.language, """
        (import_statement) @import
        (import_from_statement) @import
        """)
        cursor = QueryCursor(query)
        captures = cursor.captures(root)
        for _, nodes in captures.items():
            for node in nodes:
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
            if child.type in ['identifier', 'typed_parameter', 'default_parameter']:
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
                if left.type == 'identifier':
                    return self._get_text(left, code)
                for child in left.children:
                    if child.type == 'identifier':
                        return self._get_text(child, code)
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
        return code[node.start_byte:node.end_byte]
