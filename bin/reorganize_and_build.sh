#!/bin/bash
# Reorganize scheduler -> lib and rebuild protected release

set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Create reorganized source ==="
rm -rf scheduler_new
cp -r scheduler scheduler_new

# Rename internal directories
mv scheduler_new/rl_model scheduler_new/model 2>/dev/null || true
mv scheduler_new/dataset_generator scheduler_new/data 2>/dev/null || true

echo "=== Step 2: Update imports in source files ==="
# Update imports: scheduler -> lib
find scheduler_new -name "*.py" -exec sed -i '' \
    -e 's/from scheduler\./from lib./g' \
    -e 's/import scheduler\./import lib./g' \
    {} \;

# Update internal paths: rl_model -> model, dataset_generator -> data
find scheduler_new -name "*.py" -exec sed -i '' \
    -e 's/\.rl_model\./.model./g' \
    -e 's/\.dataset_generator\./.data./g' \
    {} \;

# Update tools and launchers
for f in scheduler_new/tools/*.py run_training.py; do
    if [ -f "$f" ]; then
        sed -i '' \
            -e 's/from scheduler\./from lib./g' \
            -e 's/import scheduler\./import lib./g' \
            -e 's/\.rl_model\./.model./g' \
            -e 's/\.dataset_generator\./.data./g' \
            "$f"
    fi
done

echo "=== Step 3: Rename scheduler_new to lib ==="
rm -rf lib
mv scheduler_new lib

echo "=== Step 4: Build protected release ==="
rm -rf release_new
mkdir -p release_new

# Copy lib (the renamed scheduler)
cp -r lib release_new/

# Obfuscate
echo "Obfuscating..."
python bin/obfuscate.py release_new/lib 2>/dev/null || true

# Compile to bytecode
echo "Compiling to bytecode..."
python -OO -m compileall -b -f release_new/lib 2>/dev/null || true

# Remove .py source files
find release_new/lib -name "*.py" -type f -delete

# Copy other files
cp -r configs release_new/ 2>/dev/null || true
cp -r data release_new/ 2>/dev/null || true
cp requirements.txt release_new/ 2>/dev/null || true
cp docs/README_RELEASE.md release_new/README.md 2>/dev/null || true

# Create updated launchers
cat > release_new/generate_training_config.py << 'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.tools.gen_config import main
if __name__ == "__main__": main()
EOF

cat > release_new/run_training.py << 'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.model.ablation_gnn_traj_main import main as train
# ... rest of training script
EOF

cat > release_new/eval_heuristics.py << 'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.tools.eval_heuristics import main
if __name__ == "__main__": main()
EOF

cat > release_new/eval_agents.py << 'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.tools.eval_agents import main
if __name__ == "__main__": main()
EOF

# Cleanup
rm -rf lib  # Remove temp lib from project root
find release_new -name "*.c" -delete 2>/dev/null || true
find release_new -name ".DS_Store" -delete 2>/dev/null || true
find release_new -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Done! ==="
echo "New release in: release_new/"
ls -la release_new/
