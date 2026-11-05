#!/bin/bash
# Test script to validate submit.sh override file generation logic
# This simulates submit.sh without actually submitting SLURM jobs

set -euo pipefail

echo "=========================================================================="
echo "TEST: submit.sh Override File Generation Logic"
echo "=========================================================================="
echo

# Create temporary test directory
TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

echo "Test directory: $TEST_DIR"
echo

# ========== Simulate submit.sh configuration ==========
SCFLP_INSTANCES=(
  "SCFLP/scenarios_s500_sigma0.1_0.3_cov6_merve_with_covariates_and_capacities/p50_stochastic.txt"
)

SUMND_INSTANCES=()

SCMND_INSTANCES=(
  "SUMND/R_stochastic/S100_sigma0.10-0.30_cov5/r01.2.stoch.json"
)

submit_method_categories_grid=("Exact" "ML")
submit_ml_method_types_grid=(
  "ML_Augmented_BD"
  "ML_Augmented_BD_with_duals"
)

CONFIG_ROOT="$TEST_DIR/config"
mkdir -p "$CONFIG_ROOT"

LIST_FILE="$CONFIG_ROOT/task_list.txt"
: > "$LIST_FILE"

# ========== Generate override files (exact copy of submit.sh logic) ==========
echo "Generating override files (simulating submit.sh)..."
echo

# Generate override files for SCFLP
if [[ ${#SCFLP_INSTANCES[@]} -gt 0 ]]; then
  for INSTANCE in "${SCFLP_INSTANCES[@]}"; do
    # Create safe filename from instance path
    SAFE_NAME=$(echo "$INSTANCE" | tr '/' '_' | tr '.' '_')
    
    # Exact method combination
    METHOD="Exact"
    OVERRIDE_BASENAME="config_override_SCFLP_${SAFE_NAME}_${METHOD}.py"
    OVERRIDE_FILE="$CONFIG_ROOT/$OVERRIDE_BASENAME"
    cat <<EOF > "$OVERRIDE_FILE"
problem_classes_to_instances = {
    "SCFLP": ["$INSTANCE"]
}
method_categories_grid = ["Exact"]
ml_method_types_grid = []
EOF
    echo "$OVERRIDE_BASENAME|SCFLP" >> "$LIST_FILE"
    
    # ML method combinations
    for ML_METHOD in "${submit_ml_method_types_grid[@]}"; do
      METHOD="ML_${ML_METHOD}"
      OVERRIDE_BASENAME="config_override_SCFLP_${SAFE_NAME}_${METHOD}.py"
      OVERRIDE_FILE="$CONFIG_ROOT/$OVERRIDE_BASENAME"
      cat <<EOF > "$OVERRIDE_FILE"
problem_classes_to_instances = {
    "SCFLP": ["$INSTANCE"]
}
method_categories_grid = ["ML"]
ml_method_types_grid = ["$ML_METHOD"]
EOF
      echo "$OVERRIDE_BASENAME|SCFLP" >> "$LIST_FILE"
    done
  done
  echo "✓ Generated $((${#SCFLP_INSTANCES[@]} * (1 + ${#submit_ml_method_types_grid[@]}))) SCFLP override files"
fi

# Generate override files for SCMND
if [[ ${#SCMND_INSTANCES[@]} -gt 0 ]]; then
  for INSTANCE in "${SCMND_INSTANCES[@]}"; do
    SAFE_NAME=$(echo "$INSTANCE" | tr '/' '_' | tr '.' '_')
    
    # Exact method combination
    METHOD="Exact"
    OVERRIDE_BASENAME="config_override_SCMND_${SAFE_NAME}_${METHOD}.py"
    OVERRIDE_FILE="$CONFIG_ROOT/$OVERRIDE_BASENAME"
    cat <<EOF > "$OVERRIDE_FILE"
problem_classes_to_instances = {
    "SCMND": ["$INSTANCE"]
}
method_categories_grid = ["Exact"]
ml_method_types_grid = []
EOF
    echo "$OVERRIDE_BASENAME|SCMND" >> "$LIST_FILE"
    
    # ML method combinations
    for ML_METHOD in "${submit_ml_method_types_grid[@]}"; do
      METHOD="ML_${ML_METHOD}"
      OVERRIDE_BASENAME="config_override_SCMND_${SAFE_NAME}_${METHOD}.py"
      OVERRIDE_FILE="$CONFIG_ROOT/$OVERRIDE_BASENAME"
      cat <<EOF > "$OVERRIDE_FILE"
problem_classes_to_instances = {
    "SCMND": ["$INSTANCE"]
}
method_categories_grid = ["ML"]
ml_method_types_grid = ["$ML_METHOD"]
EOF
      echo "$OVERRIDE_BASENAME|SCMND" >> "$LIST_FILE"
    done
  done
  echo "✓ Generated $((${#SCMND_INSTANCES[@]} * (1 + ${#submit_ml_method_types_grid[@]}))) SCMND override files"
fi

echo
echo "Task list file ($LIST_FILE):"
cat "$LIST_FILE"
echo

# ========== Validate override files ==========
echo "=========================================================================="
echo "Validating override files..."
echo "=========================================================================="
echo

TOTAL_TASKS=$(wc -l < "$LIST_FILE")
echo "Total tasks generated: $TOTAL_TASKS"
echo

TASK_NUM=1
while IFS='|' read -r OVERRIDE_CONFIG PROBLEM_CLASS; do
    OVERRIDE_FILE="$CONFIG_ROOT/$OVERRIDE_CONFIG"
    
    echo "Task $TASK_NUM: $OVERRIDE_CONFIG"
    echo "  Problem Class: $PROBLEM_CLASS"
    
    # Check file exists
    if [[ ! -f "$OVERRIDE_FILE" ]]; then
        echo "  ❌ ERROR: File does not exist!"
        exit 1
    fi
    
    # Extract values from override file
    PROBLEM_CLASSES=$(grep "problem_classes_to_instances = {" -A 2 "$OVERRIDE_FILE" | grep -oP '"[A-Z]+":' | head -1 | tr -d '":')
    
    # Validate using Python (quick syntax check + value validation)
    python3 << PYEOF
import sys
sys.path.insert(0, '$CONFIG_ROOT')

# Import override
override_filename = '$OVERRIDE_CONFIG'.replace('.py', '')
try:
    exec(open('$OVERRIDE_FILE').read())
except Exception as e:
    print(f"  ❌ Syntax error: {e}")
    sys.exit(1)

# Validate structure
exec(open('$OVERRIDE_FILE').read())
if 'problem_classes_to_instances' not in locals():
    print("  ❌ Missing: problem_classes_to_instances")
    sys.exit(1)
if 'method_categories_grid' not in locals():
    print("  ❌ Missing: method_categories_grid")
    sys.exit(1)
if 'ml_method_types_grid' not in locals():
    print("  ❌ Missing: ml_method_types_grid")
    sys.exit(1)

# Print values
pci = locals()['problem_classes_to_instances']
mcg = locals()['method_categories_grid']
mmg = locals()['ml_method_types_grid']

print(f"  ✓ problem_classes_to_instances: {pci}")
print(f"  ✓ method_categories_grid: {mcg}")
print(f"  ✓ ml_method_types_grid: {mmg}")

# Validate values
if 'Exact' in mcg:
    if mmg != []:
        print(f"  ❌ ERROR: Exact method should have empty ml_method_types_grid, got {mmg}")
        sys.exit(1)

if 'ML' in mcg:
    if len(mmg) != 1:
        print(f"  ❌ ERROR: ML method should have exactly 1 method, got {mmg}")
        sys.exit(1)

PYEOF
    
    if [[ $? -ne 0 ]]; then
        echo "  ❌ Validation failed"
        exit 1
    fi
    
    echo "  ✅ PASSED"
    echo
    
    ((TASK_NUM++))
done < "$LIST_FILE"

echo "=========================================================================="
echo "✅ ALL OVERRIDE FILES VALIDATED SUCCESSFULLY"
echo "=========================================================================="
echo
echo "Summary:"
echo "  - All $TOTAL_TASKS override files generated correctly"
echo "  - All files have required variables: problem_classes_to_instances, method_categories_grid, ml_method_types_grid"
echo "  - Exact tasks have empty ml_method_types_grid"
echo "  - ML tasks have exactly one method in ml_method_types_grid"
echo
echo "The submit.sh override file generation is CORRECT! ✅"
