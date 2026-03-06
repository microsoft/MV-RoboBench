#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Consolidated script for running reasoning model inference in parallel.
# Compatible with Azure OpenAI 2025-01-01-preview reasoning models (e.g., gpt-5, o4-mini).
#
# REQS:
#   - ENDPOINT_URL (required; set via environment variable)
#   - AZURE_OPENAI_API_VERSION=2025-01-01-preview (defaulted below)
#   - Entra ID auth via DefaultAzureCredential (Managed Identity or local dev sign-in)
#
# HOW TO RUN:
#   python3 run_vision_reasoning_inference.py \
#     --qa-root /path/to/QA_root \
#     --results-root /path/to/results \
#     --deployments your-deployment-1 your-deployment-2 \
#     --system-prompt-path /path/to/system.json
#
# NOTES:
#   - Reasoning models accept `max_completion_tokens`, not `max_tokens`.
#   - Reasoning models only support the default temperature (=1). Do not pass custom sampling params.
#

import os
import re
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Azure OpenAI + retry
from openai import AzureOpenAI, RateLimitError, APIConnectionError, AuthenticationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


# -------------------------- Client & Helpers -------------------------- #

def get_client() -> Tuple[AzureOpenAI, str]:
    """
    Initializes and returns the AzureOpenAI client and deployment name.
    """
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    if not endpoint:
        raise ValueError("ENDPOINT_URL is required. Please set it in your environment.")
    if not deployment:
        raise ValueError("DEPLOYMENT_NAME is required. Please set it in your environment.")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )
    return client, deployment


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages_vision(
    question: str,
    options: Dict[str, str],
    system_prompt_obj: Dict[str, Any],
    images_list: List[Dict[str, Any]],
    json_base_path: Path
) -> List[Dict[str, Any]]:
    """
    Build a message list with your QA text + (optional) images.
    We keep your external system/developer prompt object as-is in the first message.
    """
    opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    # Give the model a strict output contract to improve parse reliability.
    text_content = (
        "Choose the single best option that answers the question (A–E).\n"
        "Your response must end with a single line: 'Answer: X'.\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Options:\n"
        f"{opts_str}"
    )

    content_list = [{"type": "text", "text": text_content}]

    # MODIFIED: Loop through images and append their illustration text (if any) THEN the image data.
    for image_info in images_list or []:
        try:
            image_path = (json_base_path / image_info.get("path", "")).resolve()
            illustration_text = image_info.get("illustration")

            if image_path.is_file():
                # Add the text label for the image first, if it exists
                if illustration_text:
                    content_list.append({
                        "type": "text",
                        "text": f"Image context: {illustration_text}"
                    })
                
                # Then, add the image data
                b64 = encode_image_to_base64(image_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            else:
                if image_info.get("path"):
                    print(f"Warning: Image file not found at {image_path}")
        except Exception as e:
            print(f"Warning: Failed to attach image {image_info.get('path','N/A')}: {e}")

    # Respect your provided prompt as the first message (role may be 'system' or 'developer'; both are fine).
    return [
        system_prompt_obj,
        {"role": "user", "content": content_list},
    ]


def log_retry_attempt(retry_state):
    print(f"Rate limit hit. Retrying in {retry_state.next_action.sleep:.2f}s (Attempt {retry_state.attempt_number})")


# ---------------------------- Model Call ----------------------------- #

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, AuthenticationError)),
    # MODIFIED: Capped the max wait time between retries to 16 seconds.
    wait=wait_exponential(multiplier=2, min=2, max=16),
    # MODIFIED: Increased the total number of attempts to 10.
    stop=stop_after_attempt(10),
    before_sleep=log_retry_attempt,
)
def ask_model(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, Any]],
    *,
    max_completion_tokens: int = 4096
) -> Dict[str, Any]:
    """
    Calls reasoning model – supported parameters:
      - model, messages, max_completion_tokens
      - temperature must be default (1) so we do not pass it
      - do NOT pass top_p / penalties
    """
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        # temperature omitted (default = 1)
        # top_p / freq / presence omitted (unsupported / not needed)
        stream=False,
    )
    data = json.loads(resp.to_json())
    text = data["choices"][0]["message"].get("content", "").strip() if data.get("choices") else ""

    # Parse the final choice robustly
    m = re.search(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        explanation = text
    else:
        # Fallback: scan for a single capital letter A–E; default to A
        mf = re.search(r"\b([A-E])\b(?!\.)", text)
        choice = mf.group(1).upper() if mf else "A"
        explanation = text if len(text) <= 600 else text[:600] + "..."

    return {
        "raw_response": text,
        "parsed": {"choice": choice, "explanation": explanation},
    }


def to_safe_model_tag(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._-")
    return sanitized or "custom_model"


# ------------------------ I/O & Processing -------------------------- #

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_single_file(
    src_json_path: Path,
    qa_root: Path,
    results_root: Path,
    model_tag: str,
    key_name: str,
    max_completion_tokens: int,
    system_prompt_obj: Dict[str, Any],
) -> str:
    """
    Processes one QA.json: load → build messages → call model → write result.
    """
    client, deployment = get_client()
    rel = src_json_path.relative_to(qa_root)
    dst_json = (results_root / "reasoning_models" / model_tag / rel).resolve()

    if dst_json.exists():
        try:
            if key_name in load_json(dst_json):
                return "skipped"
        except Exception:
            pass

    try:
        obj = load_json(src_json_path)
        question = obj.get("question", "").strip()
        options = obj.get("options", {})
        images_list = obj.get("images", [])

        if not question or not isinstance(options, dict) or not options:
            obj[key_name] = {"model": deployment, "error": "Missing or invalid question/options"}
            save_json(dst_json, obj)
            return "processed_with_error"

        messages = build_messages_vision(
            question=question,
            options=options,
            system_prompt_obj=system_prompt_obj,
            images_list=images_list,
            json_base_path=src_json_path.parent
        )

        result = ask_model(
            client,
            deployment,
            messages,
            max_completion_tokens=max_completion_tokens  # pass by name to avoid positional arg issues
        )

        obj[key_name] = {
            "model": deployment,
            "choice": result["parsed"]["choice"],
            "explanation": result["parsed"]["explanation"],
        }
        save_json(dst_json, obj)
        return "processed"

    except Exception as e:
        try:
            err_obj = load_json(src_json_path)
        except Exception:
            err_obj = {}
        err_obj[key_name] = {"model": deployment, "error": str(e)}
        save_json(dst_json, err_obj)
        return "failed"


def process_folder_parallel(
    qa_root: Path,
    results_root: Path,
    model_tag: str,
    key_name: str,
    max_completion_tokens: int,
    num_workers: int,
    system_prompt_obj: Dict[str, Any],
) -> Tuple[int, int]:
    qa_files = sorted(qa_root.rglob("QA.json"))
    print(f"Found {len(qa_files)} QA.json files to process with {num_workers} workers.")

    worker_func = partial(
        process_single_file,
        qa_root=qa_root,
        results_root=results_root,
        model_tag=model_tag,
        key_name=key_name,
        max_completion_tokens=max_completion_tokens,
        system_prompt_obj=system_prompt_obj,
    )

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker_func, qa_files), total=len(qa_files), desc="Processing files")
        )

    processed_count = results.count("processed") + results.count("processed_with_error") + results.count("failed")
    skipped_count = results.count("skipped")
    return processed_count, skipped_count


# ------------------------------- Main -------------------------------- #

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Run reasoning model inference in parallel across multiple deployments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--qa-root", type=str, default=str(repo_root / "data/qa"))
    parser.add_argument("--results-root", type=str, default=str(repo_root / "results"))
    parser.add_argument("--key-name", type=str, default="reasoning_model_output")
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--system-prompt-path", type=str, default=str(repo_root / "evaluation/sysprompt/system_zeroshot.json"))
    parser.add_argument(
        "--deployments",
        nargs="+",
        required=True,
        help="Azure OpenAI deployment names to run, e.g. reasoner_a reasoner_b",
    )
    args = parser.parse_args()

    deployments_to_run = args.deployments

    system_prompt_path = Path(args.system_prompt_path)
    if not system_prompt_path.is_file():
        print(f"Error: System prompt file not found at {system_prompt_path}")
        raise SystemExit(1)
    system_prompt_obj = load_json(system_prompt_path)
    print(f"Loaded system prompt from: {system_prompt_path}")

    qa_root = Path(args.qa_root).resolve()
    results_root = Path(args.results_root).resolve()

    if not qa_root.is_dir():
        print(f"Error: qa-root does not exist: {qa_root}")
        raise SystemExit(1)

    for deployment in deployments_to_run:
        print("=" * 80)
        print(f"Starting parallel inference for reasoning deployment: {deployment}")
        print("=" * 80)

        # Make deployment visible to worker processes
        os.environ["DEPLOYMENT_NAME"] = deployment

        model_tag = to_safe_model_tag(deployment)

        results_path = results_root / "reasoning_models" / model_tag
        print(f"Results will be saved in: {results_path}")

        processed, skipped = process_folder_parallel(
            qa_root=qa_root,
            results_root=results_root,
            model_tag=model_tag,
            key_name=args.key_name,
            max_completion_tokens=args.max_completion_tokens,
            num_workers=args.num_workers,
            system_prompt_obj=system_prompt_obj,
        )

        print(f"\nFinished inference for {deployment}.")
        print(f"   Processed: {processed}, Skipped (already exists): {skipped}")
        print("-" * 80 + "\n")

    print("All reasoning model deployments have been processed.")


if __name__ == "__main__":
    main()