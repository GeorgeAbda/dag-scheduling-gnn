#!/usr/bin/env python3
"""Standalone entry point for heuristic evaluation."""
import sys
import os

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import and run the evaluation script
import runpy
runpy.run_module('scripts.eval_heuristics_multi_cases', run_name='__main__')
