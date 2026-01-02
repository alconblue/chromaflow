#!/usr/bin/env bash
# ChromaFlow Test Runner
# Usage: ./scripts/run_tests.sh [pytest args]
#
# Examples:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh -v                 # Verbose output
#   ./scripts/run_tests.sh -k "test_ingest"   # Run specific tests
#   ./scripts/run_tests.sh --cov              # With coverage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "ChromaFlow Test Suite"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(python3 --version)"
echo ""

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Warning: No virtual environment detected."
    echo "Consider running: poetry shell"
    echo ""
fi

# Default pytest args
PYTEST_ARGS=(
    "-v"
    "--tb=short"
    "-x"  # Stop on first failure
)

# Add coverage if requested
if [[ " $* " == *" --cov "* ]]; then
    PYTEST_ARGS+=(
        "--cov=src/chromaflow"
        "--cov-report=term-missing"
        "--cov-report=html:htmlcov"
    )
fi

# Run pytest
echo "Running: pytest ${PYTEST_ARGS[*]} $*"
echo "========================================"
python3 -m pytest "${PYTEST_ARGS[@]}" "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "All tests passed!"
else
    echo "Tests failed with exit code: $EXIT_CODE"
fi
echo "========================================"

exit $EXIT_CODE
