"""System prompt and LLM interaction for language feedback."""

import json
import os
from typing import Any

import httpx
from fastapi import HTTPException
from openai import AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

VALID_ERROR_TYPES = {
  "grammar",
  "spelling",
  "word_choice",
  "punctuation",
  "word_order",
  "missing_word",
  "extra_word",
  "conjugation",
  "gender_agreement",
  "number_agreement",
  "tone_register",
  "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

SYSTEM_PROMPT = """\
You are a language-learning feedback assistant for chat-based learners.

Your job:
- Analyze one learner sentence written in the target language.
- Return concise, supportive, learner-friendly corrections.

Hard rules:
1) Output must be a single valid JSON object only. No markdown, no extra text.
2) If sentence is already correct: set is_correct=true, errors=[], and corrected_sentence exactly equal to the original sentence.
3) Use minimal edits only. Preserve learner intent, voice, and wording whenever possible.
4) For each error include: original, correction, error_type, explanation.
5) explanation must be written in the learner's native language.
6) error_type must be exactly one of:
   grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other
7) difficulty must be exactly one of: A1, A2, B1, B2, C1, C2.
8) Judge difficulty by sentence complexity, not by number of mistakes.
9) Work across all writing systems (Latin and non-Latin scripts) with the same structure.

Examples:
Input: target_language=Spanish; native_language=English; sentence="Yo soy fue al mercado ayer."
Output: {"corrected_sentence":"Yo fui al mercado ayer.","is_correct":false,"errors":[{"original":"soy fue","correction":"fui","error_type":"conjugation","explanation":"You combined two verb forms. Use fui for I went in this sentence."}],"difficulty":"A2"}

Input: target_language=German; native_language=English; sentence="Ich habe gestern einen interessanten Film gesehen."
Output: {"corrected_sentence":"Ich habe gestern einen interessanten Film gesehen.","is_correct":true,"errors":[],"difficulty":"B1"}
"""

REPAIR_INSTRUCTION = """\
Your previous response was invalid. Return ONLY one valid JSON object with this exact shape and valid enum values.
Required top-level fields: corrected_sentence (string), is_correct (boolean), errors (array), difficulty (A1|A2|B1|B2|C1|C2).
Each errors item must include: original, correction, error_type, explanation.
error_type must be one of: grammar, spelling, word_choice, punctuation, word_order, missing_word, extra_word, conjugation, gender_agreement, number_agreement, tone_register, other.
No additional text outside JSON.
"""

_CACHE: dict[tuple[str, str, str], FeedbackResponse] = {}


class ProviderCallError(Exception):
  pass


class OutputValidationError(Exception):
  pass


def _extract_json_payload(content: str) -> str:
  text = content.strip()
  if text.startswith("```") and text.endswith("```"):
    lines = text.splitlines()
    if len(lines) >= 3:
      if lines[0].strip().startswith("```"):
        lines = lines[1:]
      if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
      text = "\n".join(lines).strip()
  return text


def _build_user_message(request: FeedbackRequest) -> str:
  return (
    f"Target language: {request.target_language}\n"
    f"Native language: {request.native_language}\n"
    f"Sentence: {request.sentence}"
  )


def _parse_and_validate_output(content: str, original_sentence: str) -> FeedbackResponse:
  try:
    data = json.loads(_extract_json_payload(content))
  except json.JSONDecodeError as error:
    raise OutputValidationError("Model output was not valid JSON") from error

  if not isinstance(data, dict):
    raise OutputValidationError("Top-level output must be a JSON object")

  required_fields = {"corrected_sentence", "is_correct", "errors", "difficulty"}
  missing = [field for field in required_fields if field not in data]
  if missing:
    raise OutputValidationError(f"Missing required field(s): {', '.join(missing)}")

  if not isinstance(data["corrected_sentence"], str):
    raise OutputValidationError("corrected_sentence must be a string")
  if not isinstance(data["is_correct"], bool):
    raise OutputValidationError("is_correct must be a boolean")
  if not isinstance(data["errors"], list):
    raise OutputValidationError("errors must be an array")
  if data["difficulty"] not in VALID_DIFFICULTIES:
    raise OutputValidationError("difficulty must be one of A1, A2, B1, B2, C1, C2")

  for index, error_item in enumerate(data["errors"]):
    if not isinstance(error_item, dict):
      raise OutputValidationError(f"errors[{index}] must be an object")

    error_required_fields = {"original", "correction", "error_type", "explanation"}
    error_missing = [field for field in error_required_fields if field not in error_item]
    if error_missing:
      raise OutputValidationError(
        f"errors[{index}] missing required field(s): {', '.join(error_missing)}"
      )

    if not isinstance(error_item["original"], str):
      raise OutputValidationError(f"errors[{index}].original must be a string")
    if not isinstance(error_item["correction"], str):
      raise OutputValidationError(f"errors[{index}].correction must be a string")
    if not isinstance(error_item["explanation"], str):
      raise OutputValidationError(f"errors[{index}].explanation must be a string")
    if error_item["error_type"] not in VALID_ERROR_TYPES:
      raise OutputValidationError(
        f"errors[{index}].error_type must be a valid enum value"
      )

  if data["is_correct"] is True:
    if data["errors"] != []:
      raise OutputValidationError("When is_correct is true, errors must be empty")
    if data["corrected_sentence"] != original_sentence:
      data["corrected_sentence"] = original_sentence
  else:
    if len(data["errors"]) == 0:
      raise OutputValidationError(
        "When is_correct is false, errors must contain at least one item"
      )

  try:
    return FeedbackResponse(**data)
  except Exception as error:
    raise OutputValidationError("Output failed response model validation") from error


async def _call_openai(messages: list[dict[str, str]]) -> str:
  try:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
      model=OPENAI_MODEL,
      messages=messages,
      response_format={"type": "json_object"},
      temperature=0.2,
      timeout=LLM_TIMEOUT_SECONDS,
    )
    content = response.choices[0].message.content
    if not content:
      raise ProviderCallError("OpenAI returned empty content")
    return content
  except Exception as error:
    raise ProviderCallError("OpenAI request failed") from error


async def _call_anthropic(messages: list[dict[str, str]]) -> str:
  api_key = os.getenv("ANTHROPIC_API_KEY")
  if not api_key:
    raise ProviderCallError("Anthropic API key is not configured")

  user_parts = [entry["content"] for entry in messages if entry["role"] == "user"]
  user_content = "\n\n".join(user_parts)

  payload: dict[str, Any] = {
    "model": ANTHROPIC_MODEL,
    "max_tokens": 1200,
    "temperature": 0.2,
    "system": SYSTEM_PROMPT,
    "messages": [{"role": "user", "content": user_content}],
  }

  headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
  }

  try:
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
      response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
      )
      response.raise_for_status()

    data = response.json()
    content_blocks = data.get("content", [])
    text_blocks = [
      block.get("text", "")
      for block in content_blocks
      if isinstance(block, dict) and block.get("type") == "text"
    ]
    content = "\n".join(part for part in text_blocks if part).strip()
    if not content:
      raise ProviderCallError("Anthropic returned empty content")
    return content
  except Exception as error:
    raise ProviderCallError("Anthropic request failed") from error


async def _generate_with_provider(
  provider: str, user_message: str, original_sentence: str
) -> FeedbackResponse:
  call_fn = _call_openai if provider == "openai" else _call_anthropic

  base_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_message},
  ]

  first_output = await call_fn(base_messages)
  try:
    return _parse_and_validate_output(first_output, original_sentence)
  except OutputValidationError:
    repair_messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": user_message},
      {"role": "user", "content": REPAIR_INSTRUCTION},
    ]
    second_output = await call_fn(repair_messages)
    return _parse_and_validate_output(second_output, original_sentence)


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
  cache_key = (request.sentence, request.target_language, request.native_language)
  cached = _CACHE.get(cache_key)
  if cached is not None:
    return cached.model_copy(deep=True)

  user_message = _build_user_message(request)

  has_openai = bool(os.getenv("OPENAI_API_KEY"))
  has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

  if not has_openai and not has_anthropic:
    raise HTTPException(status_code=503, detail="No LLM provider API key configured")

  providers: list[str] = []
  if has_openai:
    providers.append("openai")
  if has_anthropic:
    providers.append("anthropic")

  last_call_error: ProviderCallError | None = None

  for provider in providers:
    try:
      result = await _generate_with_provider(provider, user_message, request.sentence)
      _CACHE[cache_key] = result
      return result.model_copy(deep=True)
    except OutputValidationError:
      raise HTTPException(
        status_code=502,
        detail="LLM returned invalid response format after one retry",
      )
    except ProviderCallError as error:
      last_call_error = error
      continue

  if last_call_error is not None:
    raise HTTPException(
      status_code=503,
      detail="Feedback provider temporarily unavailable",
    ) from last_call_error

  raise HTTPException(
    status_code=503,
    detail="Unable to generate feedback at this time",
  )
