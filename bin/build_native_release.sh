#!/bin/bash
# Build native release with Cython-compiled .so files
# These CANNOT be decompiled back to Python source code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RELEASE_DIR="$PROJECT_ROOT/release_native"
BUILD_DIR="$PROJECT_ROOT/build_cython"

echo "=============================================="
echo "Building Native Release (Cython compiled)"
echo "=============================================="

# Clean previous builds
rm -rf "$RELEASE_DIR" "$BUILD_DIR"
mkdir -p "$RELEASE_DIR"

# Check for Cython
if ! python -c "import Cython" 2>/dev/null; then
    echo "Installing Cython..."
    pip install cython
fi

echo ""
echo "[1/5] Compiling Python to native .so files..."
echo "----------------------------------------------"

# Run Cython compilation
python setup_cython.py build_ext --inplace 2>&1 | tail -20

echo ""
echo "[2/5] Creating release directory structure..."
echo "----------------------------------------------"

# Create directory structure
mkdir -p "$RELEASE_DIR/bin"
mkdir -p "$RELEASE_DIR/data"
mkdir -p "$RELEASE_DIR/runs/datasets"
mkdir -p "$RELEASE_DIR/scheduler"
mkdir -p "$RELEASE_DIR/scripts"

echo ""
echo "[3/5] Copying compiled .so files..."
echo "----------------------------------------------"

# Copy compiled .so files (preserving directory structure)
find scheduler -name "*.so" -type f | while read so_file; do
    dest_dir="$RELEASE_DIR/$(dirname "$so_file")"
    mkdir -p "$dest_dir"
    cp "$so_file" "$dest_dir/"
    echo "  Copied: $so_file"
done

find scripts -name "*.so" -type f | while read so_file; do
    dest_dir="$RELEASE_DIR/$(dirname "$so_file")"
    mkdir -p "$dest_dir"
    cp "$so_file" "$dest_dir/"
    echo "  Copied: $so_file"
done

# Copy __init__.py files (needed for Python imports, but contain no code)
find scheduler -name "__init__.py" -type f | while read init_file; do
    dest_dir="$RELEASE_DIR/$(dirname "$init_file")"
    mkdir -p "$dest_dir"
    # Create empty __init__.py (no source code exposed)
    touch "$dest_dir/__init__.py"
done

find scripts -name "__init__.py" -type f | while read init_file; do
    dest_dir="$RELEASE_DIR/$(dirname "$init_file")"
    mkdir -p "$dest_dir"
    touch "$dest_dir/__init__.py"
done

echo ""
echo "[4/5] Copying data files and configs..."
echo "----------------------------------------------"

# Copy data files
cp -r data/* "$RELEASE_DIR/data/" 2>/dev/null || true
echo "  Copied: data/"

# Copy dataset seeds
cp -r runs/datasets/* "$RELEASE_DIR/runs/datasets/" 2>/dev/null || true
echo "  Copied: runs/datasets/"

# Copy shell scripts
cp bin/train_*.sh "$RELEASE_DIR/bin/" 2>/dev/null || true
cp bin/eval_*.sh "$RELEASE_DIR/bin/" 2>/dev/null || true
cp bin/run_*.sh "$RELEASE_DIR/bin/" 2>/dev/null || true
chmod +x "$RELEASE_DIR/bin/"*.sh 2>/dev/null || true
echo "  Copied: bin/*.sh"

echo ""
echo "[5/5] Updating shell scripts for native release..."
echo "----------------------------------------------"

# Update shell scripts to use the release directory
for script in "$RELEASE_DIR/bin/"*.sh; do
    if [ -f "$script" ]; then
        # Update PYTHONPATH setup
        sed -i '' 's|export PYTHONPATH=.*|export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"|g' "$script" 2>/dev/null || true
    fi
done

# Create README
cat > "$RELEASE_DIR/README.md" << 'EOF'
# Native Release (Cython Compiled)

This release contains **natively compiled** Python code that cannot be decompiled.

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

## Usage

```bash
cd release_native
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Train long-CP specialist
./bin/train_longcp_specialist.sh

# Train wide specialist  
./bin/train_wide_specialist.sh
```

## Contents

- `bin/` - Shell scripts to run experiments
- `scheduler/` - Compiled native modules (.so files)
- `scripts/` - Compiled evaluation scripts (.so files)
- `data/` - Configuration files
- `runs/datasets/` - Seed files for reproducibility

## Note

The `.so` files are platform-specific (macOS/Linux). 
This release was built for: $(uname -s) $(uname -m)
EOF

# Copy requirements
cp requirements.txt "$RELEASE_DIR/" 2>/dev/null || echo "torch>=2.0.0
torch-geometric>=2.0.0
gymnasium>=0.29.0
numpy
pandas
matplotlib
tyro
tqdm" > "$RELEASE_DIR/requirements.txt"

# Create archive
echo ""
echo "Creating archive..."
cd "$PROJECT_ROOT"
rm -f release_native.zip
zip -r release_native.zip release_native -x "*.pyc" -x "*__pycache__*" -x "*.py" -q

echo ""
echo "=============================================="
echo "Native Release Build Complete!"
echo "=============================================="
echo ""
echo "Output directory: $RELEASE_DIR"
echo "Archive: release_native.zip ($(du -h release_native.zip | cut -f1))"
echo ""
echo "Contents:"
ls -la "$RELEASE_DIR"
echo ""
echo "Compiled modules:"
find "$RELEASE_DIR" -name "*.so" | wc -l | xargs echo "  Total .so files:"
echo ""
echo "IMPORTANT: .so files are platform-specific!"
echo "This build is for: $(uname -s) $(uname -m)"
echo "=============================================="
