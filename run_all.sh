#!/bin/bash
# run_all.sh — Launch full BER evaluation across all BW configs.
# Usage: bash run_all.sh [NMC]
#   NMC: Monte Carlo trials (default 20)

NMC=${1:-20}
echo "═══ Full BER Evaluation — NMC=$NMC ═══"
echo ""

# Configs: M SCS_kHz
CONFIGS=(
    "256 15"
    "256 60"
    "512 60"
)

# Optional smaller configs (uncomment to include):
# CONFIGS+=("64 60" "128 60" "256 30")

for cfg in "${CONFIGS[@]}"; do
    read -r M SCS <<< "$cfg"
    echo "──────────────────────────────────────"
    echo "Config: M=$M SCS=${SCS}kHz"
    echo "──────────────────────────────────────"
    
    # Run each channel separately to stay within timeout
    for CH in AWGN TDL-A TDL-D TDL-C; do
        echo "  Channel: $CH"
        python3 run_eval.py $M $SCS $NMC $CH
        echo ""
    done
done

echo "═══ All evaluations complete ═══"
echo "Results in results/"
ls -la results/*.json
