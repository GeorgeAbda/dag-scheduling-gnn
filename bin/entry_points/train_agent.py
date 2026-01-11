#!/usr/bin/env python3
"""Standalone entry point for training agents."""
import sys
import os

# Add the bundled modules to path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
else:
    # Running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scheduler.rl_model.ablation_gnn_traj_main import main, Args
import tyro

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
