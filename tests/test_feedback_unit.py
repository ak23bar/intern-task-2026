"""Unit tests -- run without an API key using mocked LLM responses."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app import feedback as feedback_module
from app.feedback import get_feedback
from app.models import FeedbackRequest


def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _mock_raw_completion(raw_content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = raw_content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture(autouse=True)
def clear_feedback_cache():
    feedback_module._CACHE.clear()
    yield
    feedback_module._CACHE.clear()


@pytest.mark.asyncio
async def test_feedback_with_errors():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_retry_once_after_invalid_output_then_succeeds():
    invalid_first = _mock_raw_completion("not-json")
    valid_second = _mock_completion(
        {
            "corrected_sentence": "Yo fui al mercado ayer.",
            "is_correct": False,
            "errors": [
                {
                    "original": "soy fue",
                    "correction": "fui",
                    "error_type": "conjugation",
                    "explanation": "You combined two verb forms.",
                }
            ],
            "difficulty": "A2",
        }
    )

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            side_effect=[invalid_first, valid_second]
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert instance.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_invalid_output_after_one_retry_returns_502():
    invalid = _mock_raw_completion("{bad-json")

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(side_effect=[invalid, invalid])

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        with pytest.raises(HTTPException) as exc:
            await get_feedback(request)

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_is_correct_true_normalizes_corrected_sentence_to_input():
    normalized_response = _mock_completion(
        {
            "corrected_sentence": "Ich sah gestern einen Film.",
            "is_correct": True,
            "errors": [],
            "difficulty": "B1",
        }
    )

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=normalized_response
        )

        request = FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence
    assert instance.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_fenced_json_output_parses_without_retry():
    payload = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You combined two verb forms.",
            }
        ],
        "difficulty": "A2",
    }
    fenced = _mock_raw_completion(f"```json\n{json.dumps(payload)}\n```")

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=fenced)

        result = await get_feedback(
            FeedbackRequest(
                sentence="Yo soy fue al mercado ayer.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert instance.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_is_correct_false_requires_at_least_one_error():
    invalid_first = _mock_completion(
        {
            "corrected_sentence": "Yo fui al mercado ayer.",
            "is_correct": False,
            "errors": [],
            "difficulty": "A2",
        }
    )
    valid_second = _mock_completion(
        {
            "corrected_sentence": "Yo fui al mercado ayer.",
            "is_correct": False,
            "errors": [
                {
                    "original": "soy fue",
                    "correction": "fui",
                    "error_type": "conjugation",
                    "explanation": "You combined two verb forms.",
                }
            ],
            "difficulty": "A2",
        }
    )

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            side_effect=[invalid_first, valid_second]
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_non_latin_script_response_parses_and_validates():
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "住む uses に for location.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "grammar"
    assert "に" in result.errors[0].correction


@pytest.mark.asyncio
async def test_cache_returns_validated_response_without_second_provider_call():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            }
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )

        first = await get_feedback(request)
        second = await get_feedback(request)

    assert first.corrected_sentence == second.corrected_sentence
    assert instance.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_cache_returns_deep_copy_isolated_from_mutation():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You combined two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        request = FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )

        first = await get_feedback(request)
        first.errors[0].explanation = "mutated"
        second = await get_feedback(request)

    assert second.errors[0].explanation == "You combined two verb forms."
    assert instance.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_openai_failure_falls_back_to_anthropic():
    fallback_json = json.dumps(
        {
            "corrected_sentence": "Yo fui al mercado ayer.",
            "is_correct": False,
            "errors": [
                {
                    "original": "soy fue",
                    "correction": "fui",
                    "error_type": "conjugation",
                    "explanation": "You combined two verb forms.",
                }
            ],
            "difficulty": "A2",
        }
    )

    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "bad-openai", "ANTHROPIC_API_KEY": "good-anthropic"},
        clear=False,
    ):
        with patch(
            "app.feedback._call_openai",
            new=AsyncMock(side_effect=feedback_module.ProviderCallError("openai down")),
        ) as mock_openai:
            with patch(
                "app.feedback._call_anthropic",
                new=AsyncMock(return_value=fallback_json),
            ) as mock_anthropic:
                result = await get_feedback(
                    FeedbackRequest(
                        sentence="Yo soy fue al mercado ayer.",
                        target_language="Spanish",
                        native_language="English",
                    )
                )

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert mock_openai.await_count == 1
    assert mock_anthropic.await_count == 1


@pytest.mark.asyncio
async def test_both_providers_fail_returns_503():
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "bad-openai", "ANTHROPIC_API_KEY": "bad-anthropic"},
        clear=False,
    ):
        with patch(
            "app.feedback._call_openai",
            new=AsyncMock(side_effect=feedback_module.ProviderCallError("openai down")),
        ):
            with patch(
                "app.feedback._call_anthropic",
                new=AsyncMock(side_effect=feedback_module.ProviderCallError("anthropic down")),
            ):
                with pytest.raises(HTTPException) as exc:
                    await get_feedback(
                        FeedbackRequest(
                            sentence="Yo soy fue al mercado ayer.",
                            target_language="Spanish",
                            native_language="English",
                        )
                    )

    assert exc.value.status_code == 503
    assert "temporarily unavailable" in exc.value.detail


@pytest.mark.asyncio
async def test_invalid_field_type_triggers_retry_then_502():
    invalid_type = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": "not-a-list",
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            side_effect=[
                _mock_completion(invalid_type),
                _mock_completion(invalid_type),
            ]
        )

        with pytest.raises(HTTPException) as exc:
            await get_feedback(
                FeedbackRequest(
                    sentence="Yo soy fue al mercado ayer.",
                    target_language="Spanish",
                    native_language="English",
                )
            )

    assert exc.value.status_code == 502
    assert instance.chat.completions.create.await_count == 2


@pytest.mark.parametrize("error_type", sorted(feedback_module.VALID_ERROR_TYPES))
@pytest.mark.asyncio
async def test_all_error_types_are_accepted(error_type: str):
    mock_response = {
        "corrected_sentence": "Texto corregido.",
        "is_correct": False,
        "errors": [
            {
                "original": "bad",
                "correction": "good",
                "error_type": error_type,
                "explanation": "Mensaje breve.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        result = await get_feedback(
            FeedbackRequest(
                sentence="Texto malo.",
                target_language="Spanish",
                native_language="English",
            )
        )

    assert result.errors[0].error_type == error_type
    assert result.difficulty == "A2"


@pytest.mark.parametrize("difficulty", sorted(feedback_module.VALID_DIFFICULTIES))
@pytest.mark.asyncio
async def test_all_difficulty_levels_are_accepted(difficulty: str):
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": difficulty,
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )

        result = await get_feedback(
            FeedbackRequest(
                sentence="Ich habe gestern einen Film gesehen.",
                target_language="German",
                native_language="English",
            )
        )

    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty == difficulty
