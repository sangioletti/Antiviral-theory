#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python tests/run_all.py "$@"
