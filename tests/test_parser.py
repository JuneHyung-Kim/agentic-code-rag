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
    assert parser.python_parser is not None
    assert parser.c_parser is not None
    assert parser.cpp_parser is not None


def test_parse_python_function():
    """Test parsing a simple Python function"""
    parser = CodeParser()
    code = '''
    def hello_world(name):
        """Say hello"""
        print(f"Hello, {name}!")
    '''
    nodes = parser.parse_file("sample.py", code)
    func_nodes = [n for n in nodes if n.type == 'function' and n.name == 'hello_world']
    assert len(func_nodes) == 1
    assert 'def hello_world' in func_nodes[0].content
    assert func_nodes[0].docstring == "Say hello"
    assert any('name' in arg for arg in func_nodes[0].arguments)


def test_parse_python_class():
    """Test parsing a Python class"""
    parser = CodeParser()
    code = '''
    class Calculator:
        def add(self, a, b):
            return a + b
    '''
    nodes = parser.parse_file("sample.py", code)
    class_nodes = [n for n in nodes if n.type == 'class' and n.name == 'Calculator']
    method_nodes = [n for n in nodes if n.type == 'method' and n.name == 'add']
    assert len(class_nodes) == 1
    assert len(method_nodes) == 1
    assert method_nodes[0].parent_name == 'Calculator'


def test_parse_python_decorated_function():
    """Test parsing a decorated Python function"""
    parser = CodeParser()
    code = '''
    @decorator
    def decorated(x):
        return x
    '''
    nodes = parser.parse_file("sample.py", code)
    assert any(n.type == 'function' and n.name == 'decorated' for n in nodes)


def test_parse_c_function():
    """Test parsing a C function"""
    parser = CodeParser()
    code = '''
    int add(int a, int b) {
        return a + b;
    }
    '''
    nodes = parser.parse_file("sample.c", code)
    func_nodes = [n for n in nodes if n.type == 'function' and n.name == 'add']
    assert len(func_nodes) == 1


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
    nodes = parser.parse_file("sample.cpp", code)
    class_nodes = [n for n in nodes if n.type == 'class' and n.name == 'Vector']
    method_nodes = [n for n in nodes if n.name == 'push_back']
    assert len(class_nodes) == 1
    assert len(method_nodes) == 1
    assert method_nodes[0].parent_name == 'Vector'


def test_parse_cpp_qualified_method():
    """Test parsing a qualified C++ method"""
    parser = CodeParser()
    code = '''
    class Vector {
    public:
        void pop();
    };

    void Vector::pop() {}
    '''
    nodes = parser.parse_file("sample.cpp", code)
    pop_nodes = [n for n in nodes if n.name.endswith('pop')]
    assert len(pop_nodes) >= 1
    assert any(n.parent_name == 'Vector' for n in pop_nodes)


def test_parse_empty_code():
    """Test parsing empty code"""
    parser = CodeParser()
    nodes = parser.parse_file("sample.py", '')
    assert len(nodes) == 0


def test_parse_invalid_language():
    """Test parsing with unsupported extension"""
    parser = CodeParser()
    nodes = parser.parse_file("sample.txt", 'code')
    assert len(nodes) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
