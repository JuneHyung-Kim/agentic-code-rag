from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python
import tree_sitter_c
import tree_sitter_cpp

class CodeParser:
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self._initialize_parsers()

    def _initialize_parsers(self):
        # Python
        py_lang = Language(tree_sitter_python.language())
        py_parser = Parser(py_lang)
        self.parsers['python'] = py_parser
        self.languages['python'] = py_lang
        
        # C
        c_lang = Language(tree_sitter_c.language())
        c_parser = Parser(c_lang)
        self.parsers['c'] = c_parser
        self.languages['c'] = c_lang

        # C++
        cpp_lang = Language(tree_sitter_cpp.language())
        cpp_parser = Parser(cpp_lang)
        self.parsers['cpp'] = cpp_parser
        self.languages['cpp'] = cpp_lang

    def parse(self, code: str, language: str):
        """
        Parse code string into an AST.
        """
        if language not in self.parsers:
            raise ValueError(f"Language {language} not supported.")
        
        tree = self.parsers[language].parse(bytes(code, "utf8"))
        return tree

    def extract_definitions(self, code: str, language: str):
        """
        Extract function and class definitions using Tree-sitter queries.
        """
        tree = self.parse(code, language)
        root_node = tree.root_node
        
        definitions = []
        
        if language == 'python':
            query_scm = """
            (function_definition) @function
            (class_definition) @class
            """
        elif language == 'c':
            query_scm = """
            (function_definition) @function
            (struct_specifier) @struct
            """
        elif language == 'cpp':
            query_scm = """
            (function_definition) @function
            (class_specifier) @class
            (struct_specifier) @struct
            """
        else:
            return []

        # Get the language object to create the query
        ts_language = self.languages[language]
        
        try:
            query = Query(ts_language, query_scm)
        except Exception as e:
            print(f"Query error for {language}: {e}")
            return []

        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)

        # Flatten captures dict to list of (node, name)
        all_captures = []
        for name, nodes in captures.items():
            for node in nodes:
                all_captures.append((node, name))
        
        # Sort captures by start byte to handle them in order
        sorted_captures = sorted(all_captures, key=lambda x: x[0].start_byte)

        for node, name in sorted_captures:
            definitions.append({
                'type': name,
                'name': self._get_node_name(node, code),
                'start_line': node.start_point[0],
                'end_line': node.end_point[0],
                'content': code[node.start_byte:node.end_byte]
            })
            
        return definitions

    def _get_node_name(self, node, code: str) -> str:
        # Helper to extract name identifier from a node
        child_by_field = node.child_by_field_name('name')
        if child_by_field:
            return code[child_by_field.start_byte:child_by_field.end_byte]
        
        # Fallback for C++ where name might be complex or nested
        for child in node.children:
            if child.type in ['identifier', 'type_identifier', 'field_identifier']:
                 return code[child.start_byte:child.end_byte]
            # Handle C++ qualified names (Namespace::Class)
            if child.type == 'qualified_identifier':
                return code[child.start_byte:child.end_byte]
                
        return "anonymous"
