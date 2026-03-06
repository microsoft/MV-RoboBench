#!/usr/bin/env bash
set -euo pipefail

# ---- Editable Defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# IMPORTANT: Do not commit real endpoints/secrets.
export ENDPOINT_URL="${ENDPOINT_URL:-}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2025-01-01-preview}"
QA_ROOT="${QA_ROOT:-${REPO_ROOT}/data/qa}"
BASE_RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
KEY_NAME="${KEY_NAME:-zeroshot_output}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-${REPO_ROOT}/evaluation/sysprompt/system_zeroshot.json}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
PY="${PY:-python3}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SAVE_MESSAGES="${SAVE_MESSAGES:-false}"
TRUNCATE_MESSAGE_CHARS="${TRUNCATE_MESSAGE_CHARS:-8000}"
# ---------------------------

# ---- List of models to run in sequence ----
DEPLOYMENTS_TO_RUN=(
  "your-deployment-name-1"
  "your-deployment-name-2"
)
# -------------------------------------------

if [[ -z "${ENDPOINT_URL}" ]]; then
  echo "Error: ENDPOINT_URL is required. Set it before running this script."
  exit 1
fi

if [[ ! -f "${SYSTEM_PROMPT_PATH}" ]]; then
  echo "Error: SYSTEM_PROMPT_PATH does not exist: ${SYSTEM_PROMPT_PATH}"
  exit 1
fi

if [[ ! -d "${QA_ROOT}" ]]; then
  echo "Error: QA_ROOT does not exist: ${QA_ROOT}"
  exit 1
fi

# Install dependencies if they are not present
pip show openai >/dev/null 2>&1 || pip install --upgrade "openai>=1.51.0"
pip show azure-identity >/dev/null 2>&1 || pip install --upgrade "azure-identity>=1.17.1"
pip show tqdm >/dev/null 2>&1 || pip install --upgrade "tqdm"
pip show tenacity >/dev/null 2>&1 || pip install --upgrade "tenacity"

# --- Main execution loop ---
for deployment in "${DEPLOYMENTS_TO_RUN[@]}"; do
  echo "================================================================="
  echo "Starting Zero-Shot (Direct) inference for: ${deployment}"
  echo "================================================================="

  export DEPLOYMENT_NAME="${deployment}"

  # Keep only safe filename chars for result folder names.
  model_tag="$(echo "${deployment}" | tr -c '[:alnum:]_.-' '_')"

  results_path="${BASE_RESULTS_ROOT}"
  echo "Base results directory: ${results_path}"

  ${PY} "${SCRIPT_DIR}/run_zero_shot_inference.py" \
    --qa-root "${QA_ROOT}" \
    --results-root "${results_path}" \
    --model-tag "${model_tag}" \
    --key-name "${KEY_NAME}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --num-workers "${NUM_WORKERS}" \
    --system-prompt-path "${SYSTEM_PROMPT_PATH}" \
    --save-messages "${SAVE_MESSAGES}" \
    --truncate-message-chars "${TRUNCATE_MESSAGE_CHARS}"

  echo "Finished inference for ${deployment}."
  echo ""
done

echo "All deployments have been processed."