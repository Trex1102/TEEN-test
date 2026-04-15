#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-cifar100}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/.anaconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/.anaconda/etc/profile.d/conda.sh"
else
  source ~/.bashrc
fi

conda activate fscil-env
cd "${REPO_ROOT}"
bash scripts/run_teen_cpr_qpr.sh "${DATASET}" "$@"
