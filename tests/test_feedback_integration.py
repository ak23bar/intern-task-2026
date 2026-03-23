"""Integration tests -- require an LLM provider API key to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no OpenAI or Anthropic key is available.
"""

import os

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
    reason="No LLM provider API key set -- skipping integration tests",
)

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


def _assert_contract_safe(result):
    assert isinstance(result.corrected_sentence, str)
    assert isinstance(result.is_correct, bool)
    assert isinstance(result.errors, list)
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert isinstance(error.original, str)
        assert isinstance(error.correction, str)
        assert isinstance(error.explanation, str)


def _assert_behavior_invariants(result, original_sentence: str):
    if result.is_correct:
        assert result.errors == []
        assert result.corrected_sentence == original_sentence
    else:
        assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_spanish_error():
    original_sentence = "Yo soy fue al mercado ayer."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)
    for error in result.errors:
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_correct_german():
    original_sentence = "Ich habe gestern einen interessanten Film gesehen."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == original_sentence
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_french_gender_errors():
    original_sentence = "La chat noir est sur le table."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_japanese_particle():
    original_sentence = "私は東京を住んでいます。"
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any("に" in e.correction for e in result.errors)
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_portuguese_conjugation_error_realistic():
    original_sentence = "Eu está muito feliz com o resultado."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Portuguese",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type in {"conjugation", "grammar"} for e in result.errors)
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_chinese_word_order_error_realistic():
    original_sentence = "我去了学校昨天。"
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Chinese",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type in {"word_order", "grammar"} for e in result.errors)
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_correct_spanish_sentence_invariant():
    original_sentence = "Ayer fui al mercado con mi hermana."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Spanish",
            native_language="English",
        )
    )

    assert result.is_correct is True
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_russian_case_error_realistic():
    original_sentence = "Я живу в Москва."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Russian",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_korean_conjugation_error_realistic():
    original_sentence = "저는 한국어를 공부에요."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Korean",
            native_language="English",
        )
    )

    assert result.is_correct is False
    assert len(result.errors) >= 1
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)


@pytest.mark.asyncio
async def test_correct_japanese_sentence_invariant():
    original_sentence = "私は東京に住んでいます。"
    result = await get_feedback(
        FeedbackRequest(
            sentence=original_sentence,
            target_language="Japanese",
            native_language="English",
        )
    )

    assert result.is_correct is True
    _assert_contract_safe(result)
    _assert_behavior_invariants(result, original_sentence)
