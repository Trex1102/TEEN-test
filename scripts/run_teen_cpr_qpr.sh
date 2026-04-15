#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-cifar100}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_ROOT}"
python scripts/teen_cpr_qpr_eval.py --dataset "${DATASET}" "$@"
