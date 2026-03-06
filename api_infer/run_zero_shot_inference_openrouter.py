#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Script for running zero-shot vision inference via OpenRouter.
#
# REQUIREMENTS:
#   - OPENROUTER_API_KEY (required; set via environment variable)
#   - Image paths in your QA.json files can be either public URLs
#     (e.g., http://...) or local relative paths (e.g., ./view1.png).
#
# OPTIONAL ENV VARS:
#   - OPENROUTER_BASE_URL: Defaults to https://openrouter.ai/api/v1
#   - YOUR_SITE_URL: Sets the HTTP-Referer header for OpenRouter analytics.
#   - YOUR_SITE_NAME: Sets the X-Title header for OpenRouter analytics.
#
# HOW TO RUN:
#   python3 run_openrouter_zero_shot_inference.py \
#     --qa-root /path/to/your/QA_root \
#     --results-root /path/to/results \
#     --model openrouter/model-name \
#     --system-prompt-path /path/to/system_zeroshot.json
#

import os
import re
import json
import base64
import mimetypes
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Use the standard OpenAI library
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------------------------- Client & Helpers -------------------------- #

def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client configured for OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required. Please set it in your environment.")

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def to_safe_model_tag(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("._-")
    return sanitized or "custom_model"

def encode_image_to_base64(image_path: Path) -> str:
    """
    Reads an image file and returns it as a base64 encoded data URI.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{encoded_string}"


def build_messages_vision(
    question: str,
    options: Dict[str, str],
    system_prompt_obj: Dict[str, Any],
    images_list: List[Dict[str, Any]],
    base_path: Path
) -> List[Dict[str, Any]]:
    """
    Builds the message list for a vision model.
    Handles both public URLs and local file paths.
    """
    opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    text_content = (
        "Choose the single best option that answers the question (A–E).\n"
        "Your response must end with a single line: 'Answer: X'.\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Options:\n"
        f"{opts_str}"
    )

    content_list = [{"type": "text", "text": text_content}]

    for image_info in images_list or []:
        image_path_str = image_info.get("path", "")
        
        if image_path_str.startswith(("http://", "https://")):
            image_url = image_path_str
        else:
            try:
                full_image_path = (base_path / image_path_str).resolve()
                image_url = encode_image_to_base64(full_image_path)
            except FileNotFoundError as e:
                print(f"Warning: Skipping image. File not found: {e}")
                continue
            except Exception as e:
                print(f"Warning: Skipping image due to encoding error for {image_path_str}: {e}")
                continue

        illustration_text = image_info.get("illustration")
        if illustration_text:
            content_list.append({
                "type": "text",
                "text": f"Image context: {illustration_text}"
            })

        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    
    # Standard format: System prompt is the first message in the list.
    return [
        system_prompt_obj,
        {"role": "user", "content": content_list}
    ]


def log_retry_attempt(retry_state):
    print(f"API call failed. Retrying in {retry_state.next_action.sleep:.2f}s (Attempt {retry_state.attempt_number})")


# ---------------------------- Model Call ----------------------------- #

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, AuthenticationError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=log_retry_attempt,
)
def ask_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float
) -> Dict[str, Any]:
    """
    Calls a model via OpenRouter using the standard messages format.
    """
    extra_headers = {
        "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),
        "X-Title": os.getenv("YOUR_SITE_NAME", ""),
    }

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # The 'system' keyword argument is removed as it caused the error.
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={k: v for k, v in extra_headers.items() if v},
    )

    data = json.loads(resp.to_json())
    text = data["choices"][0]["message"].get("content", "").strip() if data.get("choices") else ""

    m = re.search(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        explanation = text
    else:
        mf = re.search(r"\b([A-E])\b(?!\.)", text)
        choice = mf.group(1).upper() if mf else "A"
        explanation = text if len(text) <= 600 else text[:600] + "..."

    return {
        "raw_response": text,
        "parsed": {"choice": choice, "explanation": explanation},
    }


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
    model: str,
    key_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt_obj: Dict[str, Any],
) -> str:
    """
    Processes one QA.json file.
    """
    client = get_client()
    model_tag = model.replace("/", "_")
    rel = src_json_path.relative_to(qa_root)
    dst_json = (results_root / model_tag / rel).resolve()

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
            obj[key_name] = {"model": model, "error": "Missing or invalid question/options"}
            save_json(dst_json, obj)
            return "processed_with_error"

        messages = build_messages_vision(
            question=question,
            options=options,
            system_prompt_obj=system_prompt_obj,
            images_list=images_list,
            base_path=src_json_path.parent
        )

        result = ask_model(
            client,
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        obj[key_name] = {
            "model": model,
            "choice": result["parsed"]["choice"],
            "explanation": result["parsed"]["explanation"],
        }
        save_json(dst_json, obj)
        return "processed"

    except Exception as e:
        print(f"Error processing {src_json_path}: {e}")
        try:
            err_obj = load_json(src_json_path)
        except Exception:
            err_obj = {}
        err_obj[key_name] = {"model": model, "error": str(e)}
        save_json(dst_json, err_obj)
        return "failed"


def process_folder_parallel(
    qa_root: Path,
    results_root: Path,
    model: str,
    key_name: str,
    max_tokens: int,
    temperature: float,
    num_workers: int,
    system_prompt_obj: Dict[str, Any],
) -> Tuple[int, int]:
    qa_files = sorted(qa_root.rglob("QA.json"))
    print(f"Found {len(qa_files)} QA.json files to process for model '{model}' with {num_workers} workers.")

    worker_func = partial(
        process_single_file,
        qa_root=qa_root,
        results_root=results_root,
        model=model,
        key_name=key_name,
        max_tokens=max_tokens,
        temperature=temperature,
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
        description="Run OpenRouter zero-shot inference in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--qa-root", type=str, default=str(repo_root / "data/qa"), help="Root directory containing QA.json files.")
    parser.add_argument("--results-root", type=str, default=str(repo_root / "results/openrouter"), help="Directory to save results.")
    parser.add_argument("--system-prompt-path", type=str, default=str(repo_root / "evaluation/sysprompt/system_zeroshot.json"), help="Path to system prompt JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Model name on OpenRouter, e.g. anthropic/claude-3.5-sonnet.")
    parser.add_argument("--key-name", type=str, default="openrouter_output", help="Key to store results under in the JSON.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model.")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel processes.")
    args = parser.parse_args()

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

    model_tag = to_safe_model_tag(args.model)
    results_path = results_root / model_tag
    print(f"Results will be saved in: {results_path}")

    processed, skipped = process_folder_parallel(
        qa_root=qa_root,
        results_root=results_root,
        model=args.model,
        key_name=args.key_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.num_workers,
        system_prompt_obj=system_prompt_obj,
    )

    print(f"\nFinished inference for {args.model}.")
    print(f"   Processed: {processed}, Skipped (already exists): {skipped}")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()

