# Evaluation Inference Scripts

This folder provides parallel inference scripts for multiple backends:

- Azure OpenAI zero-shot text inference
- Azure OpenAI reasoning model inference (vision-capable)
- OpenRouter zero-shot vision inference

All scripts are written for batch processing over `QA.json` files.

## Folder Structure

```text
evaluation/
	api_infer/
		run_zero_shot_inference.py
		run_zero_shot_inference.sh
		run_zero_shot_inference_reasoning.py
		run_zero_shot_inference_openrouter.py
	sysprompt/
		system_zeroshot.json
		system_zeroshot_wexplaination.json
```

## Input Data Format

Scripts search recursively for files named `QA.json` under `--qa-root`.

Minimum expected fields:

```json
{
	"question": "...",
	"options": {
		"A": "...",
		"B": "...",
		"C": "...",
		"D": "...",
		"E": "..."
	}
}
```

For vision scripts, optional image inputs are supported:

```json
{
	"images": [
		{
			"path": "./view1.png",
			"illustration": "front camera view"
		}
	]
}
```

`path` can be a local relative path or an `http(s)` URL (OpenRouter script supports both).

## Security Notes

- Do not hardcode keys, endpoints, or local absolute paths in scripts.
- Provide credentials only via environment variables.
- Do not commit result files containing sensitive prompts/data.
- Keep `--save-messages` disabled unless you explicitly need prompt logging.

## Dependencies

Install dependencies in your Python environment:

```bash
pip install -U openai azure-identity tenacity tqdm
```

## 1) Azure OpenAI Zero-Shot (Text)

Script: `api_infer/run_zero_shot_inference.py`

Required environment variables:

- `ENDPOINT_URL`
- `DEPLOYMENT_NAME`

Optional:

- `AZURE_OPENAI_API_VERSION` (default: `2025-01-01-preview`)

Example:

```bash
export ENDPOINT_URL="https://<your-resource>.openai.azure.com/"
export DEPLOYMENT_NAME="<your-deployment-name>"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"

python3 evaluation/api_infer/run_zero_shot_inference.py \
	--qa-root /path/to/qa \
	--results-root /path/to/results \
	--model-tag custom_model \
	--key-name zeroshot_output \
	--system-prompt-path evaluation/sysprompt/system_zeroshot.json \
	--num-workers 10 \
	--save-messages false
```

### Batch Wrapper (Optional)

Script: `api_infer/run_zero_shot_inference.sh`

Edit `DEPLOYMENTS_TO_RUN` in the script, then run:

```bash
bash evaluation/api_infer/run_zero_shot_inference.sh
```

## 2) Azure OpenAI Reasoning Models (Vision)

Script: `api_infer/run_zero_shot_inference_reasoning.py`

Required environment variables:

- `ENDPOINT_URL`

Optional:

- `AZURE_OPENAI_API_VERSION` (default: `2025-01-01-preview`)

Required argument:

- `--deployments` (one or more deployment names)

Example:

```bash
export ENDPOINT_URL="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"

python3 evaluation/api_infer/run_zero_shot_inference_reasoning.py \
	--qa-root /path/to/qa \
	--results-root /path/to/results \
	--deployments reasoner_deploy_a reasoner_deploy_b \
	--system-prompt-path evaluation/sysprompt/system_zeroshot.json \
	--num-workers 10
```

Notes:

- This script uses `max_completion_tokens` (reasoning model API).
- It does not pass custom sampling parameters for reasoning models.

## 3) OpenRouter Zero-Shot (Vision)

Script: `api_infer/run_zero_shot_inference_openrouter.py`

Required environment variables:

- `OPENROUTER_API_KEY`

Optional environment variables:

- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `YOUR_SITE_URL` (used as `HTTP-Referer` header)
- `YOUR_SITE_NAME` (used as `X-Title` header)

Required argument:

- `--model` (e.g. `anthropic/claude-3.5-sonnet`)

Example:

```bash
export OPENROUTER_API_KEY="<your-openrouter-key>"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

python3 evaluation/api_infer/run_zero_shot_inference_openrouter.py \
	--qa-root /path/to/qa \
	--results-root /path/to/results/openrouter \
	--model anthropic/claude-3.5-sonnet \
	--system-prompt-path evaluation/sysprompt/system_zeroshot.json \
	--num-workers 10
```

## Outputs

- Azure zero-shot text: `<results-root>/blind_models/<model-tag>/.../QA.json`
- Azure reasoning: `<results-root>/reasoning_models/<deployment-tag>/.../QA.json`
- OpenRouter: `<results-root>/<model-tag>/.../QA.json`

Each processed file adds a result object under `--key-name` (for example `zeroshot_output` or `openrouter_output`).

## Troubleshooting

- `System prompt file not found`: verify `--system-prompt-path`.
- `qa-root does not exist`: verify `--qa-root`.
- Auth errors: verify required environment variables and cloud account login.
- Missing Python imports in editor: install dependencies into the active environment.
