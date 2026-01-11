#!/usr/bin/env python3
import sys, os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from cogito.tools.eval_heuristics import main
if __name__ == "__main__": main()
