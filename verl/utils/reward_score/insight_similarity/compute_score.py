"""LM-judge reward for insight anticipation training."""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from google import genai

MODEL_NAME = os.environ.get("INSIGHT_SIMILARITY_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = int(os.environ.get("INSIGHT_SIMILARITY_MAX_TOKENS", "8192"))
DEBUG_DIR = os.environ.get("INSIGHT_SIMILARITY_DEBUG_DIR")

GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 32,
    "candidate_count": 1,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
}


def _debug_root() -> Path | None:
    if not DEBUG_DIR:
        return None
    root = Path(DEBUG_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_debug_file(prefix: str, body: str) -> None:
    root = _debug_root()
    if root is None:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (root / f"{prefix}_{timestamp}.txt").write_text(body, encoding="utf-8")


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        return genai.Client()

    raise RuntimeError(
        "Insight similarity scoring requires either GEMINI_API_KEY/GOOGLE_API_KEY "
        "or a Vertex AI setup via GOOGLE_CLOUD_PROJECT plus credentials."
    )


def _extract_tagged_text(text: str, tag: str) -> str:
    matches = re.findall(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else ""


def _extract_score(text: str) -> float | None:
    rating = _extract_tagged_text(text, "rating")
    if not rating:
        return None
    try:
        return float(rating)
    except ValueError:
        return None


def _extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    text_parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text.strip())
    return "\n\n".join(text_parts)


def _build_prompt(predicted_insight: str, reference_insight: str) -> str:
    return (
        "Below is a research insight:\n"
        "<research_insight>\n"
        f"{reference_insight}\n"
        "</research_insight>\n"
        "Below is a statement you need to evaluate:\n"
        "<statement>\n"
        f"{predicted_insight}\n"
        "</statement>\n"
        "Task: Rate how similar the statement is to the research insight (1-10).\n\n"
        "STRICT RULES:\n"
        "- Similarity requires matching the same core idea.\n"
        "- 'Inspired by', 'motivated by', or 'reasonable extension' does not count as the same idea.\n"
        "- Shared topic or keywords alone do not count as similarity.\n\n"
        "Compare explicitly:\n"
        "1) Key mechanism or method\n"
        "2) Causal logic or workflow\n"
        "3) Primary contribution or novelty\n\n"
        "Downgrade if the statement:\n"
        "- Omits the central mechanism\n"
        "- Generalizes or abstracts the insight\n"
        "- Proposes a new framework or direction\n\n"
        "Scale:\n"
        "1-2: Unrelated.\n"
        "3-4: Shares topic but not the actual insight.\n"
        "5-6: Partial conceptual overlap with at least one missing core mechanism or assumption.\n"
        "7-8: Strong match with minor differences in mechanisms or assumptions.\n"
        "9: Near-identical conceptual, causal, and motivational mapping.\n"
        "10: Same idea, same mechanism, same objective, same assumptions.\n\n"
        "Output format:\n"
        "<think>\n"
        "Briefly justify the score.\n"
        "</think>\n"
        "<rating>a number between 1 and 10</rating>\n"
    )


def _generate_score(prompt: str) -> str:
    response = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=GENERATION_CONFIG,
    )
    return _extract_response_text(response)


async def compute_score(data_source, solution_str, ground_truth, extra_info):
    del data_source, extra_info

    try:
        predicted_insight = _extract_tagged_text(solution_str, "insight").replace("**", "")
        if not predicted_insight:
            return 0.0

        prompt = _build_prompt(predicted_insight=predicted_insight, reference_insight=ground_truth)
        completion = await asyncio.to_thread(_generate_score, prompt)
        _write_debug_file("insight_similarity_prompt", f"{prompt}\n\n===== COMPLETION =====\n\n{completion}")

        rating = _extract_score(completion)
        return rating if rating is not None else 0.0
    except Exception as exc:
        _write_debug_file("insight_similarity_error", f"{type(exc).__name__}: {exc}")
        return 0.0
