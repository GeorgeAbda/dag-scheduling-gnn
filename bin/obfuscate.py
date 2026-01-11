#!/usr/bin/env python3
"""
Python code obfuscator for release protection.
- Removes comments and docstrings
- Renames local variables and private functions
- Shortens parameter names
"""
import ast
import sys
import re
import hashlib
from pathlib import Path

# Names that must NEVER be renamed
PRESERVE = {
    # Python builtins
    'True', 'False', 'None', 'self', 'cls', 'super', 'type',
    'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
    'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'bytes',
    'open', 'input', 'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr',
    'min', 'max', 'sum', 'abs', 'round', 'pow', 'divmod',
    'any', 'all', 'iter', 'next', 'callable', 'repr', 'hash', 'id', 'ord', 'chr',
    'dir', 'vars', 'globals', 'locals', 'eval', 'exec', 'compile',
    'staticmethod', 'classmethod', 'property',
    'Exception', 'BaseException', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'StopIteration', 'FileNotFoundError', 'OSError',
    'ImportError', 'ModuleNotFoundError', 'NameError', 'ZeroDivisionError',
    # Common imports (module names)
    'os', 'sys', 're', 'json', 'yaml', 'csv', 'math', 'random', 'time', 'datetime',
    'pathlib', 'Path', 'typing', 'collections', 'itertools', 'functools', 'operator',
    'copy', 'pickle', 'logging', 'argparse', 'subprocess', 'shutil', 'glob',
    'numpy', 'np', 'pandas', 'pd', 'torch', 'nn', 'F', 'plt', 'matplotlib',
    'tqdm', 'dataclass', 'field', 'dataclasses',
    # Dunder names
    '__init__', '__new__', '__del__', '__repr__', '__str__', '__bytes__',
    '__format__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
    '__hash__', '__bool__', '__getattr__', '__setattr__', '__delattr__',
    '__getattribute__', '__dir__', '__get__', '__set__', '__delete__',
    '__call__', '__len__', '__length_hint__', '__getitem__', '__setitem__',
    '__delitem__', '__iter__', '__next__', '__reversed__', '__contains__',
    '__add__', '__sub__', '__mul__', '__matmul__', '__truediv__', '__floordiv__',
    '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__',
    '__and__', '__xor__', '__or__', '__neg__', '__pos__', '__abs__', '__invert__',
    '__enter__', '__exit__', '__await__', '__aiter__', '__anext__',
    '__name__', '__main__', '__file__', '__doc__', '__dict__', '__module__',
    '__class__', '__slots__', '__all__', '__version__',
    # Common method names that might be called externally
    'main', 'run', 'start', 'stop', 'close', 'reset', 'step', 'forward', 'backward',
    'fit', 'predict', 'transform', 'train', 'eval', 'test', 'load', 'save',
    'get', 'set', 'add', 'remove', 'update', 'clear', 'copy', 'keys', 'values', 'items',
    'append', 'extend', 'insert', 'pop', 'index', 'count', 'sort', 'reverse',
    'read', 'write', 'seek', 'tell', 'flush', 'readline', 'readlines', 'writelines',
    'encode', 'decode', 'split', 'join', 'strip', 'replace', 'find', 'format',
    'lower', 'upper', 'startswith', 'endswith', 'isdigit', 'isalpha',
}

# Global counter for name generation
_counter = 0

def _gen_name(prefix='_v'):
    """Generate short obfuscated name."""
    global _counter
    chars = 'abcdefghijklmnopqrstuvwxyz'
    n = _counter
    _counter += 1
    result = []
    while True:
        result.append(chars[n % 26])
        n = n // 26 - 1
        if n < 0:
            break
    return prefix + ''.join(reversed(result))


class NameObfuscator(ast.NodeTransformer):
    """
    Disabled for safety - parameter renaming breaks keyword arguments.
    Only docstring removal is active.
    """
    pass


class DocstringRemover(ast.NodeTransformer):
    """Remove docstrings from AST."""
    
    def _remove_docstring(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, (ast.Str, ast.Constant))):
            if isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
            else:
                node.body = node.body[1:] or [ast.Pass()]
        return node
    
    def visit_Module(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)
    
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)


def remove_comments(source):
    """Remove comments while preserving strings."""
    lines = source.split('\n')
    result = []
    in_multiline = False
    multiline_char = None
    
    for line in lines:
        if in_multiline:
            # Look for end of multiline string
            end_pos = line.find(multiline_char)
            if end_pos >= 0:
                in_multiline = False
                result.append(line)
            else:
                result.append(line)
            continue
        
        new_line = []
        i = 0
        in_string = False
        string_char = None
        
        while i < len(line):
            c = line[i]
            
            if not in_string:
                # Check for string start
                if c in '"\'':
                    if line[i:i+3] in ('"""', "'''"):
                        # Check if it ends on same line
                        end = line.find(line[i:i+3], i+3)
                        if end < 0:
                            in_multiline = True
                            multiline_char = line[i:i+3]
                        new_line.append(line[i:])
                        break
                    else:
                        in_string = True
                        string_char = c
                elif c == '#':
                    break  # Rest is comment
            else:
                if c == string_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
            
            new_line.append(c)
            i += 1
        
        result.append(''.join(new_line).rstrip())
    
    return '\n'.join(result)


def obfuscate_file(filepath):
    """Obfuscate a single Python file."""
    global _counter
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Skip if too short or looks like __init__
        if len(source.strip()) < 50:
            return True
        
        # Remove comments
        source = remove_comments(source)
        
        try:
            # Parse AST
            tree = ast.parse(source)
            
            # Remove docstrings
            remover = DocstringRemover()
            tree = remover.visit(tree)
            
            # Rename variables (reset counter per file for consistency)
            _counter = 0
            renamer = NameObfuscator()
            tree = renamer.visit(tree)
            
            ast.fix_missing_locations(tree)
            
            # Verify it compiles
            compile(tree, str(filepath), 'exec')
            
            # Convert back to source
            if hasattr(ast, 'unparse'):
                source = ast.unparse(tree)
            
        except SyntaxError as e:
            # Keep comment-stripped version if AST fails
            pass
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(source)
        
        return True
        
    except Exception as e:
        print(f"  Warning: {filepath.name}: {e}")
        return False


def process_directory(directory, verbose=False):
    """Process all Python files in directory."""
    directory = Path(directory)
    success = 0
    failed = 0
    
    for pyfile in directory.rglob('*.py'):
        if pyfile.name == '__init__.py':
            continue
        
        if obfuscate_file(pyfile):
            success += 1
            if verbose:
                print(f"  Obfuscated: {pyfile.name}")
        else:
            failed += 1
    
    return success, failed


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python obfuscate.py <directory> [--verbose]")
        sys.exit(1)
    
    target = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print(f"Obfuscating Python files in: {target}")
    success, failed = process_directory(target, verbose)
    print(f"  Success: {success}, Failed: {failed}")
