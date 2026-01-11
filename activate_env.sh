#!/bin/bash
# Activate the virtual environment and set up scheduler path

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH to include scheduler (adjust this path to your scheduler location)
# Example: export PYTHONPATH="/path/to/parent/of/scheduler:$PYTHONPATH"
# Uncomment and modify the line below:
# export PYTHONPATH="/Users/anashattay/Documents/GitHub/DaDiL:$PYTHONPATH"

echo "✅ Virtual environment activated"
echo "📦 Python: $(which python)"
echo "📍 Location: $(pwd)"
echo ""
echo "⚠️  Remember to set PYTHONPATH to your scheduler location:"
echo "   export PYTHONPATH=\"/path/to/parent/of/scheduler:\$PYTHONPATH\""
echo ""
echo "🚀 Ready to run experiments!"
