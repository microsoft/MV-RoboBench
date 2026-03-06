#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from openai import AzureOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def get_client() -> Tuple[AzureOpenAI, str]:
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


def build_messages(question: str, options: Dict[str, str], system_prompt_obj: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    MODIFIED: This function now builds a standard zero-shot prompt.
    It directly asks the model for the final answer without requiring reasoning.
    """
    opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    # MODIFIED: This is the standard Zero-Shot instruction.
    user_content = (
        "Based on the question and options provided, please choose the single best answer.\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Options:\n"
        f"{opts_str}\n\n"
        "END your response with a single line: 'Answer: X', where X is one of the option letters (A-E)."
    )

    return [
        system_prompt_obj,
        {"role": "user", "content": user_content},
    ]


def log_retry_attempt(retry_state):
    print(f"Rate limit hit. Retrying in {retry_state.next_action.sleep:.2f} seconds... (Attempt {retry_state.attempt_number})")


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=log_retry_attempt,
)
def ask_model(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=False,
    )
    data = json.loads(resp.to_json())
    text = data["choices"][0]["message"].get("content", "").strip()

    # This regex is still useful for reliably extracting the final answer.
    m = re.search(r"Answer:\s*([A-E])\b", text, re.IGNORECASE | re.DOTALL)
    if m:
        choice = m.group(1).upper()
        explanation = text
    else:
        # Fallback in case the model doesn't follow the format perfectly.
        m_fallback = re.search(r"\b([A-E])\b", text)
        choice = m_fallback.group(1).upper() if m_fallback else "A"
        explanation = text if len(text) < 300 else text[:300] + "..."

    return {
        "raw_response": text,
        "parsed": {"choice": choice, "explanation": explanation},
    }


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_messages_for_logging(
    messages: List[Dict[str, Any]],
    truncate_chars: int
) -> List[Dict[str, Any]]:
    safe = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        if not isinstance(content, str):
            content = str(content)

        if truncate_chars is not None and truncate_chars > 0 and len(content) > truncate_chars:
            content = content[:truncate_chars] + f"... [truncated to {truncate_chars} chars]"

        safe.append({
            "role": m.get("role", ""),
            "content": content,
        })
    return safe


def process_single_file(
    src_json_path: Path,
    qa_root: Path,
    results_root: Path,
    model_tag: str,
    key_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt_obj: Dict[str, str],
    save_messages: bool,
    truncate_message_chars: int,
) -> str:
    client, deployment = get_client()
    rel = src_json_path.relative_to(qa_root)
    # MODIFIED: Store results in a "blind_models" subdirectory to distinguish them from CoT runs.
    dst_json = (results_root / "blind_models" / model_tag / rel).resolve()

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

        if not question or not isinstance(options, dict) or not options:
            obj[key_name] = {"model": deployment, "error": "Missing or invalid question/options"}
            save_json(dst_json, obj)
            return "processed_with_error"

        messages = build_messages(question, options, system_prompt_obj)

        input_messages_logged = None
        if save_messages:
            input_messages_logged = sanitize_messages_for_logging(messages, truncate_message_chars)

        result = ask_model(client, deployment, messages, max_tokens, temperature)

        obj[key_name] = {
            "model": deployment,
            "choice": result["parsed"].get("choice"),
            "explanation": result["parsed"].get("explanation"),
        }

        if save_messages and input_messages_logged is not None:
            obj[key_name]["input_messages"] = input_messages_logged

        obj[key_name]["leak_flags"] = {
            "has_correct_field": "correct" in obj,
        }

        save_json(dst_json, obj)
        return "processed"

    except Exception as e:
        err_obj = {}
        try:
            err_obj = load_json(src_json_path)
        except Exception:
            pass
        err_obj[key_name] = {"model": deployment, "error": str(e)}
        save_json(dst_json, err_obj)
        return "failed"


def process_folder_parallel(
    qa_root: Path,
    results_root: Path,
    model_tag: str,
    key_name: str,
    max_tokens: int,
    temperature: float,
    num_workers: int,
    system_prompt_obj: Dict[str, str],
    save_messages: bool,
    truncate_message_chars: int,
) -> Tuple[int, int]:
    qa_files = sorted(qa_root.rglob("QA.json"))
    print(f"Found {len(qa_files)} QA.json files to process with {num_workers} workers.")

    worker_func = partial(
        process_single_file,
        qa_root=qa_root,
        results_root=results_root,
        model_tag=model_tag,
        key_name=key_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt_obj=system_prompt_obj,
        save_messages=save_messages,
        truncate_message_chars=truncate_message_chars,
    )

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker_func, qa_files), total=len(qa_files), desc="Processing files")
        )

    processed_count = results.count("processed") + results.count("processed_with_error") + results.count("failed")
    skipped_count = results.count("skipped")

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="Run model inference in parallel over QA folders.")
    parser.add_argument("--qa-root", type=str, required=True, help="Path to the subtask root that contains QAxxx folders.")
    parser.add_argument("--results-root", type=str, required=True, help="Base path for results.")
    parser.add_argument("--model-tag", type=str, required=True, help="Result subdirectory name, e.g. gpt3/gpt4/custom_model.")
    parser.add_argument("--key-name", type=str, required=True, help="New key to add into each JSON.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel processes for inference.")
    parser.add_argument("--system-prompt-path", type=str, required=True, help="Path to system prompt JSON file.")
    parser.add_argument("--save-messages", type=lambda s: s.lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--truncate-message-chars", type=int, default=8000)

    args = parser.parse_args()

    system_prompt_path = Path(args.system_prompt_path)
    if not system_prompt_path.is_file():
        print(f"❌ Error: System prompt file not found at {system_prompt_path}")
        exit(1)
    system_prompt_obj = load_json(system_prompt_path)
    print(f"✅ Loaded system prompt from: {system_prompt_path}")

    qa_root, results_root = Path(args.qa_root).resolve(), Path(args.results_root).resolve()

    processed, skipped = process_folder_parallel(
        qa_root=qa_root,
        results_root=results_root,
        model_tag=args.model_tag,
        key_name=args.key_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.num_workers,
        system_prompt_obj=system_prompt_obj,
        save_messages=args.save_messages,
        truncate_message_chars=args.truncate_message_chars,
    )

    # MODIFIED: Updated print statement to reflect the new output path
    print(f"\nDone. Processed: {processed}, Skipped (existing): {skipped}")
    print(f"Results base: {results_root / 'blind_models' / args.model_tag}")


if __name__ == "__main__":
    main()