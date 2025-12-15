# pybrain.fewshot_openai
# Few-shot prompt builder + OpenAI call helper (streaming and non-streaming).
#
# Expected examples JSONL format (one JSON object per line):
#   {"input": "...", "output": "..."}

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


def load_jsonl_examples(path_to_examples_jsonl: str) -> List[Dict[str, str]]:
    path = Path(path_to_examples_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"examples jsonl not found: {path}")

    examples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" not in obj or "output" not in obj:
                raise ValueError(f"example line {lineno} must contain 'input' and 'output' keys")
            examples.append({"input": str(obj["input"]), "output": str(obj["output"])})
    if not examples:
        raise ValueError("no examples found in jsonl")
    return examples


def build_few_shot_prompt(
    general_purpose: str,
    examples: List[Dict[str, str]],
    task_input: str,
) -> str:
    parts: List[str] = []
    parts.append(general_purpose.strip())
    parts.append("")
    parts.append("Complete the below Output in adherence with the following examples:")
    parts.append("")

    for i, ex in enumerate(examples, start=1):
        parts.append(f"##### EXAMPLE {i} #####")
        parts.append("")
        parts.append("Input:")
        parts.append(ex["input"].rstrip())
        parts.append("")
        parts.append("Output:")
        parts.append(ex["output"].rstrip())
        parts.append("")
    parts.append("##### END EXAMPLES #####")
    parts.append("")
    parts.append("Below is the task at hand.")
    parts.append("")
    parts.append(f"Input: {task_input}")
    parts.append("Output:")
    return "\n".join(parts)


def few_shot_openai(
    path_to_examples_jsonl: str,
    task_input: str,
    general_purpose: str,
    model: str = "gpt-5-mini",
    max_examples: Optional[int] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[str, str, str]:
    """Returns: (model_output_text, prompt_used, response_id)."""
    examples = load_jsonl_examples(path_to_examples_jsonl)
    if max_examples is not None:
        examples = examples[: max_examples]

    prompt = build_few_shot_prompt(general_purpose, examples, task_input)

    client = client or OpenAI()
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text, prompt, resp.id


def few_shot_openai_streaming(
    path_to_examples_jsonl: str,
    task_input: str,
    general_purpose: str,
    model: str = "gpt-5-mini",
    max_examples: Optional[int] = None,
    client: Optional[OpenAI] = None,
    stream_to_console: bool = True,
) -> Tuple[str, str, str]:
    """Streaming: prints output_text deltas while accumulating full output."""
    examples = load_jsonl_examples(path_to_examples_jsonl)
    if max_examples is not None:
        examples = examples[: max_examples]

    prompt = build_few_shot_prompt(general_purpose, examples, task_input)

    client = client or OpenAI()

    chunks: List[str] = []
    response_id: str = ""

    stream = client.responses.create(
        model=model,
        input=prompt,
        stream=True,
    )

    for event in stream:
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

        if etype == "response.created":
            resp = getattr(event, "response", None) or (event.get("response") if isinstance(event, dict) else None)
            if resp is not None:
                response_id = getattr(resp, "id", None) or (resp.get("id") if isinstance(resp, dict) else "") or response_id

        elif etype == "response.output_text.delta":
            delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None) or ""
            if delta:
                chunks.append(delta)
                if stream_to_console:
                    print(delta, end="", flush=True)

        elif etype in ("response.completed", "response.failed", "response.incomplete"):
            pass

    if stream_to_console and chunks:
        print()

    return "".join(chunks), prompt, response_id
