import re
from tree_sitter import Language, Parser, Query, QueryCursor, Node
from typing import List, Optional, Set
from ..schema import CodeNode

class CppParser:
    def __init__(self, parser: Parser, language: Language, lang_type: str = 'cpp'):
        self.parser = parser
        self.language = language
        self.lang_type = lang_type  # 'c' or 'cpp'
        self._node_types = self._load_node_types()

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

    def parse(self, code: str, file_path: str) -> List[CodeNode]:
        tree = self.parser.parse(bytes(code, "utf8"))
        nodes = []
        
        # 1. Extract Includes
        includes = self._extract_includes(tree.root_node, code)
        
        # 2. Define Query
        query_parts = []
        if self._has_node_type("function_definition"):
            query_parts.append("(function_definition) @function")
        if self._has_node_type("struct_specifier"):
            query_parts.append("(struct_specifier) @struct")
        if self._has_node_type("enum_specifier"):
            query_parts.append("(enum_specifier) @enum")
        if self._has_node_type("type_definition"):
            query_parts.append("(type_definition) @typedef")
        if self._has_node_type("preproc_def"):
            query_parts.append("(preproc_def) @macro")
        if self._has_node_type("preproc_function_def"):
            query_parts.append("(preproc_function_def) @macro")
        if self._has_node_type("declaration"):
            query_parts.append("(declaration) @declaration")
        if self.lang_type == 'cpp' and self._has_node_type("class_specifier"):
            query_parts.append("(class_specifier) @class")

        if not query_parts:
            return nodes

        query = Query(self.language, "\n".join(query_parts))
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        
        capture_list = []
        for node, name in self._iter_captures(captures):
            capture_list.append((node, name))
        
        capture_list.sort(key=lambda x: x[0].start_byte)
        processed_ids = set()
        
        for node, capture_type in capture_list:
            if node.id in processed_ids:
                continue
            processed_ids.add(node.id)
            
            # Skip nodes that contain parse errors to reduce noisy captures
            if getattr(node, "has_error", False):
                continue
            
            if capture_type in ['struct', 'class', 'enum'] and not self._is_type_definition(node):
                continue

            if capture_type in ['struct', 'class', 'enum', 'typedef'] and self._is_anonymous_type(node, code):
                continue

            if capture_type == 'declaration':
                if not self._is_top_level(node):
                    continue
                if self._is_function_declaration(node) and not node.child_by_field_name('body'):
                    capture_type = 'function_decl'
                else:
                    capture_type = 'global_var'

            # Defensive: ensure function captures actually have a declarator
            if capture_type in ['function', 'function_decl'] and not self._find_function_declarator(node):
                continue

            # Resolve complex names (pointers, namespaces)
            name = self._resolve_name(node, code, capture_type)
            if not self._is_valid_name(name, capture_type):
                continue
            docstring = self._extract_docstring(node, code) if capture_type in ['function', 'function_decl', 'struct', 'class', 'enum', 'typedef'] else None
            signature = self._extract_signature(node, code, capture_type) if capture_type in ['function', 'function_decl'] else None
            if capture_type in ['function', 'function_decl'] and signature is None:
                continue
            return_type = self._extract_return_type(node, code, capture_type) if capture_type in ['function', 'function_decl'] else None
            arguments = self._extract_arguments(node, code) if capture_type in ['function', 'function_decl'] else []
            parent_name = self._extract_parent_name(node, name, code, capture_type) if capture_type in ['function', 'function_decl'] else None
            
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
                parent_name=parent_name,
                imports=includes
            )
            nodes.append(code_node)
            
        return nodes

    def _extract_includes(self, root: Node, code: str) -> List[str]:
        includes = []
        if not self._has_node_type("preproc_include"):
            return includes
        query = Query(self.language, "(preproc_include) @include")
        cursor = QueryCursor(query)
        captures = cursor.captures(root)
        for node, _ in self._iter_captures(captures):
            text = self._get_text(node, code).strip()
            if text and text not in includes:
                includes.append(text)
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
                elif decl.type in ['operator_name', 'operator_cast', 'conversion_function_id']:
                    # Handle operator overloads (e.g., operator(), operator+, casts)
                    return self._get_text(decl, code)
                else:
                    # Fallback or deeper nesting
                    if not decl.child_by_field_name('declarator'):
                         # Try to find identifier in children
                         for child in decl.children:
                             if child.type == 'identifier':
                                 return self._get_text(child, code)
                         # Try operator token in children
                         for child in decl.children:
                             if child.type in ['operator_name', 'operator_cast', 'conversion_function_id']:
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

    def _extract_parent_name(self, node: Node, name: str, code: str, capture_type: str) -> Optional[str]:
        if capture_type not in ['function', 'function_decl']:
            return None
        if "::" in name:
            return name.rsplit("::", 1)[0]

        parent = node.parent
        while parent:
            if parent.type in ['class_specifier', 'struct_specifier']:
                name_node = parent.child_by_field_name('name')
                if name_node:
                    return self._get_text(name_node, code)
            parent = parent.parent
        return None

    def _extract_signature(self, node: Node, code: str, capture_type: str) -> Optional[str]:
        if capture_type == 'function':
            body = node.child_by_field_name('body')
            if body:
                return code[node.start_byte:body.start_byte].strip()
        if capture_type == 'function_decl':
            text = self._get_text(node, code).strip()
            # Guard against runaway captures that span many lines
            if text.count("\n") > 20 or len(text) > 2000:
                return None
            return text.rstrip(';').strip()
        return None

    def _extract_return_type(self, node: Node, code: str, capture_type: str) -> Optional[str]:
        if capture_type in ['function', 'function_decl']:
            name_node = self._find_declarator_name_node(node)
            if name_node:
                return code[node.start_byte:name_node.start_byte].strip()
        type_node = node.child_by_field_name('type')
        if type_node:
            return self._get_text(type_node, code).strip()
        return None

    def _find_declarator_name_node(self, node: Node) -> Optional[Node]:
        decl = node.child_by_field_name('declarator')
        while decl:
            if decl.type == 'function_declarator':
                decl = decl.child_by_field_name('declarator')
                continue
            if decl.type in ['pointer_declarator', 'reference_declarator']:
                decl = decl.child_by_field_name('declarator')
                continue
            if decl.type in ['identifier', 'field_identifier', 'qualified_identifier', 'type_identifier']:
                return decl
            next_decl = decl.child_by_field_name('declarator')
            if not next_decl:
                break
            decl = next_decl
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

    def _is_type_definition(self, node: Node) -> bool:
        body = node.child_by_field_name('body')
        if body:
            return True
        for child in node.children:
            if child.type in ['field_declaration_list', 'enumerator_list']:
                return True
        return False

    def _is_anonymous_type(self, node: Node, code: str) -> bool:
        name_node = node.child_by_field_name('name')
        return name_node is None or self._get_text(name_node, code).strip() == ""

    def _get_text(self, node: Node, code: str) -> str:
        if not node:
            return ""
        return code[node.start_byte:node.end_byte]

    def _is_valid_name(self, name: str, capture_type: str) -> bool:
        if not name:
            return False
        stripped = name.strip()
        if not stripped:
            return False
        # Reject obvious junk: whitespace, braces, delimiters
        if any(ch.isspace() for ch in stripped):
            return False
        if any(ch in stripped for ch in "{};()[]"):
            return False
        # Allow identifiers, namespaces, template params, destructors, operators
        op_pattern = r'^operator.*$'
        ident_pattern = r'^[A-Za-z_~][A-Za-z0-9_:<>]*$'
        if re.match(op_pattern, stripped):
            return True
        if re.match(ident_pattern, stripped):
            return True
        return False

    def _iter_captures(self, captures):
        if isinstance(captures, dict):
            for name, nodes in captures.items():
                for node in nodes:
                    yield node, name
        else:
            for node, name in captures:
                yield node, name
