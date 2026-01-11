#!/usr/bin/env bash
# =============================================================================
# Build Standalone Executables
# =============================================================================
# This script compiles the Python training/evaluation code into standalone
# binary executables that don't require the source code to run.
#
# Requirements:
#   pip install pyinstaller
#
# Usage:
#   ./bin/build_executables.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=============================================="
echo "Building Standalone Executables"
echo "=============================================="

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Create entry point scripts for PyInstaller
mkdir -p "$PROJECT_ROOT/bin/entry_points"

# Entry point for training
cat > "$PROJECT_ROOT/bin/entry_points/train_agent.py" << 'EOF'
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
EOF

# Entry point for heuristic evaluation
cat > "$PROJECT_ROOT/bin/entry_points/eval_heuristics.py" << 'EOF'
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
EOF

# Entry point for agent evaluation
cat > "$PROJECT_ROOT/bin/entry_points/eval_agents.py" << 'EOF'
#!/usr/bin/env python3
"""Standalone entry point for agent evaluation."""
import sys
import os

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import runpy
runpy.run_module('scripts.eval_hetero_agents_all_cases', run_name='__main__')
EOF

echo "Building train_agent executable..."
pyinstaller \
    --onefile \
    --name train_agent \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --specpath "$BUILD_DIR" \
    --hidden-import=scheduler \
    --hidden-import=scheduler.rl_model \
    --hidden-import=scheduler.rl_model.ablation_gnn \
    --hidden-import=scheduler.rl_model.ablation_gnn_traj_main \
    --hidden-import=scheduler.dataset_generator \
    --hidden-import=scheduler.config \
    --hidden-import=torch \
    --hidden-import=numpy \
    --hidden-import=tyro \
    --hidden-import=gymnasium \
    --hidden-import=tqdm \
    --add-data "$PROJECT_ROOT/data:data" \
    --add-data "$PROJECT_ROOT/runs/datasets:runs/datasets" \
    "$PROJECT_ROOT/bin/entry_points/train_agent.py"

echo "Building eval_heuristics executable..."
pyinstaller \
    --onefile \
    --name eval_heuristics \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --specpath "$BUILD_DIR" \
    --hidden-import=scheduler \
    --hidden-import=numpy \
    --hidden-import=tyro \
    --hidden-import=tqdm \
    --add-data "$PROJECT_ROOT/data:data" \
    --add-data "$PROJECT_ROOT/runs/datasets:runs/datasets" \
    "$PROJECT_ROOT/bin/entry_points/eval_heuristics.py"

echo "Building eval_agents executable..."
pyinstaller \
    --onefile \
    --name eval_agents \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --specpath "$BUILD_DIR" \
    --hidden-import=scheduler \
    --hidden-import=torch \
    --hidden-import=numpy \
    --hidden-import=tyro \
    --hidden-import=tqdm \
    --add-data "$PROJECT_ROOT/data:data" \
    --add-data "$PROJECT_ROOT/runs/datasets:runs/datasets" \
    "$PROJECT_ROOT/bin/entry_points/eval_agents.py"

echo "=============================================="
echo "Build complete!"
echo "Executables saved to: $DIST_DIR/"
echo ""
echo "Available executables:"
ls -la "$DIST_DIR/"
echo "=============================================="
