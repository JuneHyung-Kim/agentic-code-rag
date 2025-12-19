"""
Unit tests for code parser
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexing.parser import CodeParser


def test_parser_initialization():
    """Test parser can be initialized"""
    parser = CodeParser()
    assert parser is not None
    assert 'python' in parser.parsers
    assert 'c' in parser.parsers
    assert 'cpp' in parser.parsers


def test_parse_python_function():
    """Test parsing a simple Python function"""
    parser = CodeParser()
    code = '''
def hello_world():
    """Say hello"""
    print("Hello, World!")
'''
    definitions = parser.extract_definitions(code, 'python')
    assert len(definitions) == 1
    assert definitions[0]['type'] == 'function'
    assert definitions[0]['name'] == 'hello_world'
    assert 'def hello_world' in definitions[0]['content']


def test_parse_python_class():
    """Test parsing a Python class"""
    parser = CodeParser()
    code = '''
class Calculator:
    def add(self, a, b):
        return a + b
'''
    definitions = parser.extract_definitions(code, 'python')
    # Should find class and method
    assert len(definitions) >= 1
    class_def = [d for d in definitions if d['type'] == 'class'][0]
    assert class_def['name'] == 'Calculator'


def test_parse_c_function():
    """Test parsing a C function"""
    parser = CodeParser()
    code = '''
int add(int a, int b) {
    return a + b;
}
'''
    definitions = parser.extract_definitions(code, 'c')
    assert len(definitions) == 1
    assert definitions[0]['type'] == 'function'
    assert definitions[0]['name'] == 'add'


def test_parse_cpp_class():
    """Test parsing a C++ class"""
    parser = CodeParser()
    code = '''
class Vector {
public:
    void push_back(int value) {
        // implementation
    }
};
'''
    definitions = parser.extract_definitions(code, 'cpp')
    assert len(definitions) >= 1
    class_def = [d for d in definitions if d['type'] == 'class'][0]
    assert class_def['name'] == 'Vector'


def test_parse_empty_code():
    """Test parsing empty code"""
    parser = CodeParser()
    definitions = parser.extract_definitions('', 'python')
    assert len(definitions) == 0


def test_parse_invalid_language():
    """Test parsing with unsupported language"""
    parser = CodeParser()
    with pytest.raises(ValueError):
        parser.parse('code', 'invalid_lang')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
