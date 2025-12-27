from tree_sitter import Language, Parser, Query, QueryCursor, Node
from typing import List, Optional, Set
from ..schema import CodeNode

class CppParser:
    def __init__(self, parser: Parser, language: Language, lang_type: str = 'cpp'):
        self.parser = parser
        self.language = language
        self.lang_type = lang_type  # 'c' or 'cpp'

    def parse(self, code: str, file_path: str) -> List[CodeNode]:
        tree = self.parser.parse(bytes(code, "utf8"))
        nodes = []
        
        # 1. Extract Includes
        includes = self._extract_includes(tree.root_node, code)
        
        # 2. Define Query
        query_scm = """
        (function_definition) @function
        (struct_specifier) @struct
        (enum_specifier) @enum
        (type_definition) @typedef
        (preproc_def) @macro
        (declaration) @declaration
        """
        if self.lang_type == 'cpp':
            query_scm += "\n(class_specifier) @class"
            
        query = Query(self.language, query_scm)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        
        capture_list = []
        for name, captured_nodes in captures.items():
            for node in captured_nodes:
                capture_list.append((node, name))
        
        capture_list.sort(key=lambda x: x[0].start_byte)
        processed_ids = set()
        
        for node, capture_type in capture_list:
            if node.id in processed_ids:
                continue
            processed_ids.add(node.id)
            
            if capture_type == 'declaration':
                if not self._is_top_level(node):
                    continue
                if self._is_function_declaration(node):
                    capture_type = 'function_decl'
                else:
                    capture_type = 'global_var'

            # Resolve complex names (pointers, namespaces)
            name = self._resolve_name(node, code, capture_type)
            docstring = self._extract_docstring(node, code) if capture_type in ['function', 'function_decl', 'struct', 'class', 'enum', 'typedef'] else None
            signature = self._extract_signature(node, code, capture_type) if capture_type in ['function', 'function_decl'] else None
            return_type = self._extract_return_type(node, code, capture_type) if capture_type in ['function', 'function_decl'] else None
            arguments = self._extract_arguments(node, code) if capture_type in ['function', 'function_decl'] else []
            
            code_node = CodeNode(
                type=capture_type,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                content=self._get_text(node, code),
                language=self.lang_type,
                docstring=docstring,
                arguments=arguments,
                return_type=return_type,
                signature=signature,
                imports=includes
            )
            nodes.append(code_node)
            
        return nodes

    def _extract_includes(self, root: Node, code: str) -> List[str]:
        includes = []
        # Preprocessor includes
        query = Query(self.language, "(preproc_include) @include")
        cursor = QueryCursor(query)
        captures = cursor.captures(root)
        for _, nodes in captures.items():
            for node in nodes:
                includes.append(self._get_text(node, code).strip())
        return includes

    def _extract_docstring(self, node: Node, code: str) -> Optional[str]:
        # C/C++ often uses comments right above the function
        comments = []
        prev = node.prev_sibling
        while prev:
            if prev.type == 'comment':
                comments.insert(0, self._get_text(prev, code).strip())
                prev = prev.prev_sibling
            elif prev.type in ['\n', ' ', '\r']:
                prev = prev.prev_sibling
            else:
                break
        return "\n".join(comments) if comments else None

    def _resolve_name(self, node: Node, code: str, capture_type: str) -> str:
        # Handle declarators, pointers, references, etc.
        if capture_type in ['function', 'function_decl'] or node.type == 'function_definition':
            decl = node.child_by_field_name('declarator')
            while decl:
                if decl.type == 'function_declarator':
                    decl = decl.child_by_field_name('declarator')
                elif decl.type in ['pointer_declarator', 'reference_declarator']:
                    decl = decl.child_by_field_name('declarator')
                elif decl.type in ['identifier', 'field_identifier', 'qualified_identifier', 'type_identifier']:
                    return self._get_text(decl, code)
                else:
                    # Fallback or deeper nesting
                    if not decl.child_by_field_name('declarator'):
                         # Try to find identifier in children
                         for child in decl.children:
                             if child.type == 'identifier':
                                 return self._get_text(child, code)
                    break

        if capture_type in ['global_var', 'typedef']:
            decl = node.child_by_field_name('declarator')
            while decl:
                if decl.type in ['identifier', 'field_identifier', 'qualified_identifier', 'type_identifier']:
                    return self._get_text(decl, code)
                next_decl = decl.child_by_field_name('declarator')
                if not next_decl:
                    break
                decl = next_decl
        
        # Struct/Class name
        name_node = node.child_by_field_name('name')
        if name_node:
            return self._get_text(name_node, code)

        if capture_type == 'macro':
            return self._extract_macro_name(node, code)
            
        return "anonymous"

    def _extract_signature(self, node: Node, code: str, capture_type: str) -> Optional[str]:
        if capture_type == 'function':
            body = node.child_by_field_name('body')
            if body:
                return code[node.start_byte:body.start_byte].strip()
        if capture_type == 'function_decl':
            text = self._get_text(node, code).strip()
            return text.rstrip(';').strip()
        return None

    def _extract_return_type(self, node: Node, code: str, capture_type: str) -> Optional[str]:
        type_node = node.child_by_field_name('type')
        if type_node:
            return self._get_text(type_node, code).strip()
        return None

    def _extract_arguments(self, node: Node, code: str) -> List[str]:
        params = []
        declarator = self._find_function_declarator(node)
        if not declarator:
            return params
        param_list = declarator.child_by_field_name('parameters')
        if not param_list:
            return params
        for child in param_list.children:
            if child.type == 'parameter_declaration':
                params.append(self._get_text(child, code).strip())
        return params

    def _find_function_declarator(self, node: Node) -> Optional[Node]:
        decl = node.child_by_field_name('declarator')
        while decl:
            if decl.type == 'function_declarator':
                return decl
            next_decl = decl.child_by_field_name('declarator')
            if not next_decl:
                break
            decl = next_decl
        return None

    def _is_function_declaration(self, node: Node) -> bool:
        return self._find_function_declarator(node) is not None

    def _is_top_level(self, node: Node) -> bool:
        return node.parent is not None and node.parent.type == 'translation_unit'

    def _extract_macro_name(self, node: Node, code: str) -> str:
        text = self._get_text(node, code).strip()
        parts = text.split()
        if len(parts) >= 2:
            return parts[1]
        return "macro"

    def _get_text(self, node: Node, code: str) -> str:
        if not node:
            return ""
        return code[node.start_byte:node.end_byte]
