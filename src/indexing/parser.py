import os
from typing import List
from tree_sitter import Language, Parser
from utils.logger import logger
from .schema import CodeNode
from .parsers.python_parser import PythonParser
from .parsers.cpp_parser import CppParser

class CodeParser:
    def __init__(self):
        logger.info("Initializing CodeParser with language-specific backends")
        self._initialize_parsers()

    def _initialize_parsers(self):
        try:
            import tree_sitter_python as ts_python
            import tree_sitter_c as ts_c
            import tree_sitter_cpp as ts_cpp
        except ImportError as e:
            logger.error(f"Tree-sitter languages missing: {e}")
            raise ImportError("Please install: pip install tree-sitter-python tree-sitter-c tree-sitter-cpp")

        # Initialize Python Parser
        py_lang = Language(ts_python.language())
        py_parser = Parser(py_lang)
        self.python_parser = PythonParser(py_parser, py_lang)
        
        # Initialize C Parser
        c_lang = Language(ts_c.language())
        c_parser = Parser(c_lang)
        self.c_parser = CppParser(c_parser, c_lang, 'c')

        # Initialize C++ Parser
        cpp_lang = Language(ts_cpp.language())
        cpp_parser = Parser(cpp_lang)
        self.cpp_parser = CppParser(cpp_parser, cpp_lang, 'cpp')

    def parse_file(self, file_path: str, code: str) -> List[CodeNode]:
        """Delegate parsing to the appropriate language parser."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.py':
            return self.python_parser.parse(code, file_path)
        elif ext in ['.c', '.h']:
            return self.c_parser.parse(code, file_path)
        elif ext in ['.cpp', '.hpp', '.cc', '.cxx']:
            return self.cpp_parser.parse(code, file_path)
        else:
            logger.debug(f"Unsupported file extension: {ext}")
            return []
