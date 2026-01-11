#!/usr/bin/env bash
# =============================================================================
# Package Release Distribution
# =============================================================================
# Creates a distributable package with obfuscated Python code.
# Uses pyarmor for code protection (more reliable than PyInstaller for ML code).
#
# Usage:
#   ./bin/package_release.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RELEASE_DIR="$PROJECT_ROOT/release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "=============================================="
echo "Packaging Release Distribution"
echo "=============================================="

# Copy shell scripts
echo "Copying shell scripts..."
cp -r bin "$RELEASE_DIR/"
rm -f "$RELEASE_DIR/bin/build_executables.sh"
rm -f "$RELEASE_DIR/bin/package_release.sh"
rm -rf "$RELEASE_DIR/bin/entry_points"

# Copy data files
echo "Copying data files..."
cp -r data "$RELEASE_DIR/"

# Copy dataset seeds
echo "Copying dataset seeds..."
mkdir -p "$RELEASE_DIR/runs"
cp -r runs/datasets "$RELEASE_DIR/runs/"

# Copy requirements
echo "Copying requirements..."
cp requirements.txt "$RELEASE_DIR/"

# Create a minimal scheduler package (compiled .pyc only)
echo "Compiling Python to bytecode..."
mkdir -p "$RELEASE_DIR/scheduler"

# Compile all Python files to .pyc (ignore errors in individual files)
python -m compileall -b -f -q scheduler/ 2>/dev/null || true
find scheduler -name "*.pyc" -exec sh -c '
    src="$1"
    # Get relative path from scheduler/
    rel="${src#scheduler/}"
    # Remove __pycache__ from path
    rel=$(echo "$rel" | sed "s|__pycache__/||g")
    # Remove .cpython-*.pyc suffix and add .pyc
    rel=$(echo "$rel" | sed "s|\.cpython-[0-9]*\.pyc|.pyc|g")
    dest="'"$RELEASE_DIR"'/scheduler/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
' _ {} \;

# Copy __init__.py files as empty (needed for package structure)
find scheduler -name "__init__.py" -exec sh -c '
    src="$1"
    rel="${src#scheduler/}"
    dest="'"$RELEASE_DIR"'/scheduler/$rel"
    mkdir -p "$(dirname "$dest")"
    echo "# Package" > "$dest"
' _ {} \;

# Copy scripts folder (compiled)
echo "Compiling scripts..."
mkdir -p "$RELEASE_DIR/scripts"
python -m compileall -b -f -q scripts/ 2>/dev/null || true
find scripts -name "*.pyc" -exec sh -c '
    src="$1"
    rel="${src#scripts/}"
    rel=$(echo "$rel" | sed "s|__pycache__/||g")
    rel=$(echo "$rel" | sed "s|\.cpython-[0-9]*\.pyc|.pyc|g")
    dest="'"$RELEASE_DIR"'/scripts/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
' _ {} \;

# Update shell scripts to use python with .pyc files
echo "Updating shell scripts for bytecode execution..."

# Create a wrapper script that runs compiled code
cat > "$RELEASE_DIR/bin/run_python.sh" << 'WRAPPER'
#!/usr/bin/env bash
# Wrapper to run compiled Python modules
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
exec python "$@"
WRAPPER
chmod +x "$RELEASE_DIR/bin/run_python.sh"

# Update training scripts to use the wrapper
for script in "$RELEASE_DIR/bin/train_"*.sh; do
    if [ -f "$script" ]; then
        sed -i '' 's|\"\$SCRIPT_DIR/../dist/train_agent\"|python -m scheduler.rl_model.ablation_gnn_traj_main|g' "$script"
    fi
done

sed -i '' 's|\"\$SCRIPT_DIR/../dist/eval_heuristics\"|python scripts/eval_heuristics_multi_cases.pyc|g' "$RELEASE_DIR/bin/eval_heuristics.sh"
sed -i '' 's|\"\$SCRIPT_DIR/../dist/eval_agents\"|python scripts/eval_hetero_agents_all_cases.pyc|g' "$RELEASE_DIR/bin/eval_agents_all_cases.sh"

# Clean up __pycache__ in release
find "$RELEASE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Create release README
cat > "$RELEASE_DIR/README.md" << 'README'
# DAG-Aware Energy Scheduling - Experiment Scripts

Executable scripts for reproducing experiments from:
**"On the Role of DAG Structure in Energy-Aware Scheduling with Deep Reinforcement Learning"**

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train Long-CP specialist
./bin/train_longcp_specialist.sh

# Train Wide specialist  
./bin/train_wide_specialist.sh

# Evaluate heuristics
./bin/eval_heuristics.sh

# Evaluate agents
./bin/eval_agents_all_cases.sh
```

## Configuration

Set environment variables to customize:

```bash
HOST_SPECS_PATH=data/host_specs_homoPower.json ./bin/train_longcp_specialist.sh
SEED=42 TOTAL_TIMESTEPS=1000000 ./bin/train_wide_specialist.sh
```

## Host Specification Cases

| Case | File | Description |
|------|------|-------------|
| AL | `data/host_specs.json` | Default heterogeneous |
| HP | `data/host_specs_homoPower.json` | Homogeneous power |
| HS | `data/host_specs_homospeed.json` | Homogeneous speed |
README

# Create zip archive
echo "Creating zip archive..."
cd "$PROJECT_ROOT"
zip -r release.zip release -x "*.DS_Store"

echo "=============================================="
echo "Release package created!"
echo ""
echo "Contents:"
ls -la "$RELEASE_DIR/"
echo ""
echo "Archive: release.zip ($(du -h release.zip | cut -f1))"
echo "=============================================="
