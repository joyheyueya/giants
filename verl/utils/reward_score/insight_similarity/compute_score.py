"""LM-judge similarity reward for insight anticipation training.

This module scores a model-generated insight against a reference downstream
insight using a Gemini LM judge. The prompt below matches the one reported
in the paper (Appendix C.3 / Figure 12) verbatim.

Credentials are picked up from the environment:
  - ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``: use the public GenAI API.
  - ``GOOGLE_CLOUD_PROJECT``: use Vertex AI with default credentials.
"""

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
        "Insight similarity scoring requires either GEMINI_API_KEY / GOOGLE_API_KEY "
        "or a Vertex AI setup via GOOGLE_CLOUD_PROJECT plus default credentials."
    )


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
    (root / f"{prefix}_{timestamp}.txt").write_text(body or "", encoding="utf-8")


def _extract_tag(text: str, tag: str) -> str:
    matches = re.findall(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else ""


def get_insight(text: str) -> str:
    return _extract_tag(text, "insight")


def get_rating(text: str):
    rating = _extract_tag(text, "rating")
    if not rating:
        return "None"
    try:
        return float(rating)
    except ValueError:
        return "None"


def _extract_response_text(response: object) -> str:
    text_parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text.strip())
    if text_parts:
        return "\n\n".join(text_parts)
    return getattr(response, "text", "") or ""


def _build_prompt(solution_str: str, ground_truth: str) -> str:
    return (
        "Below is a research insight:\n"
        "<research_insight>\n"
        f"{ground_truth}\n"
        "</research_insight>\n"
        "Below is a statement you need to evaluate:\n"
        "<statement>\n"
        f"{solution_str}\n"
        "</statement>\n"
        "Task: Rate how similar the statement is to the research insight (1–10).\n\n"

        "STRICT RULES:\n"
        "- Similarity requires matching the SAME core idea.\n"
        "- 'Inspired by', 'motivated by', or 'reasonable extension' ≠ same idea.\n"
        "- Shared topic or keywords alone ≠ similarity.\n\n"

        "Compare explicitly:\n"
        "1) Key mechanism/method\n"
        "2) Causal logic/workflow\n"
        "3) Primary contribution/novelty\n\n"

        "Downgrade if the statement:\n"
        "- Omits the central mechanism\n"
        "- Generalizes/abstracts the insight\n"
        "- Proposes a new framework/direction\n\n"

        "Scale:\n"
        "1–2: Unrelated.\n"
        "3–4: Shares topic but not the actual insight.\n"
        "5–6: Partial conceptual overlap; misses at least one core mechanism or misaligns assumptions.\n"
        "7–8: Strong match with only minor differences in mechanisms or assumptions.\n"
        "9: Near-identical conceptual + causal + motivational mapping; only minor, non-substantive deviations.\n"
        "10: Perfect: same ideas, same mechanism and roles, same objective/assumptions.\n"
        "### Output Format\n"
        "Format your response as follows:\n"
        "<think>\n"
        "Explain your reasoning for the rating you chose.\n"
        "</think>\n"
        "<rating>a number between 1 and 10</rating>\n"
    )


def _generate_sync(prompt: str) -> str:
    try:
        response = _get_client().models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GENERATION_CONFIG,
        )
        return _extract_response_text(response) or "None"
    except Exception as exc:
        _write_debug_file("insight_similarity_error", f"{type(exc).__name__}: {exc}")
        return "None"


async def _generate_async(prompt: str) -> str:
    return await asyncio.to_thread(_generate_sync, prompt)


async def compute_score(data_source, solution_str, ground_truth, extra_info):
    """Return a 1\u201310 similarity score (0.0 on any failure)."""
    del data_source, extra_info

    try:
        predicted_insight = get_insight(solution_str).replace("**", "")
        if not predicted_insight:
            return 0.0

        prompt = _build_prompt(predicted_insight, ground_truth)
        completion = await _generate_async(prompt)

        _write_debug_file(
            "insight_similarity_prompt",
            f"{prompt}\n\n===== COMPLETION =====\n\n{completion}",
        )

        rating = get_rating(completion)
        if rating == "None":
            return 0.0
        return float(rating)
    except Exception as exc:
        _write_debug_file("insight_similarity_error", f"{type(exc).__name__}: {exc}")
        return 0.0
