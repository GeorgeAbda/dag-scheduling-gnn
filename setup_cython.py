"""
Cython setup script to compile Python modules to native .so binaries.
These cannot be decompiled back to readable Python code.

Usage:
    python setup_cython.py build_ext --inplace
"""
from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
import glob

# Find all Python files to compile
def find_py_files(directories, exclude_patterns=None):
    """Find all .py files in directories, excluding specified patterns."""
    exclude_patterns = exclude_patterns or []
    py_files = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    filepath = os.path.join(root, file)
                    
                    # Check exclusions
                    skip = False
                    for pattern in exclude_patterns:
                        if pattern in filepath:
                            skip = True
                            break
                    
                    if not skip:
                        py_files.append(filepath)
    
    return py_files

# Directories to compile
COMPILE_DIRS = [
    'scheduler',
    'scripts',
]

# Files/patterns to exclude from compilation
EXCLUDE_PATTERNS = [
    'setup.py',
    'test_',
    '_test.py',
    'conftest.py',
    'adversarial/',  # Has syntax issues
    'experiments/',  # Not needed for release
    'viz_results/',  # Visualization only
    '_backup',       # Backup files
]

py_files = find_py_files(COMPILE_DIRS, EXCLUDE_PATTERNS)

print(f"Found {len(py_files)} Python files to compile:")
for f in py_files[:10]:
    print(f"  - {f}")
if len(py_files) > 10:
    print(f"  ... and {len(py_files) - 10} more")

if __name__ == '__main__':
    setup(
        name='scheduler',
        ext_modules=cythonize(
            py_files,
            compiler_directives={
                'language_level': '3',
                'boundscheck': False,
                'wraparound': False,
            },
            nthreads=1,  # Sequential to avoid multiprocessing issues
        ),
        cmdclass={'build_ext': build_ext},
        packages=find_packages(),
    )
