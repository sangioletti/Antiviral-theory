#!/bin/bash
# Master script to generate all figures.
# Usage: bash run_all_figures.sh

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Generating all figures"
echo "============================================"

echo ""
echo "[1/5] Fig2: NEW_Fig2, NEW_Fig2_SI, Fig2_diff_rigidity"
python Run_Fig2_all.py

echo ""
echo "[2/5] Fig3: Fig3, Fig3_SI"
python Run_Fig3_all.py

echo ""
echo "[3/5] Fig4: Fig4, Fig4_SI"
python Run_Fig4_all.py

echo ""
echo "[4/5] Fig5: NEW_Fig5, NEW_Fig5_SI, Fig5_B_2nd_version, NEW_Fig5_C_2nd_version"
python Run_Fig5_all.py

echo ""
echo "[5/5] Fig5_2D: Fig5_2D, NEW_Fig5_2D"
python Run_Fig5_2D_all.py

echo ""
echo "============================================"
echo "  All figures generated successfully!"
echo "============================================"
