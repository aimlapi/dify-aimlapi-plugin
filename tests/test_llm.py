from types import SimpleNamespace

import pytest
from dify_plugin.errors.model import InvokeAuthorizationError, InvokeError

from models.llm.llm import AimlapiLargeLanguageModel
from provider import aimlapi_client


def test_runtime_credentials_use_fixed_endpoint_and_attribution(
    monkeypatch: pytest.MonkeyPatch,
):
    model = object.__new__(AimlapiLargeLanguageModel)
    monkeypatch.setattr(
        AimlapiLargeLanguageModel,
        "get_model_mode",
        lambda self, model_name: SimpleNamespace(value="chat"),
    )
    credentials = {"api_key": "secret-key"}

    model._update_credential("openai/gpt-4o-mini", credentials)

    assert credentials["endpoint_url"] == "https://api.aimlapi.com/v1"
    assert credentials["openai_api_key"] == "secret-key"
    assert credentials["extra_headers"] == aimlapi_client.partner_headers()


def test_invoke_maps_initial_http_error(monkeypatch: pytest.MonkeyPatch):
    model = object.__new__(AimlapiLargeLanguageModel)
    monkeypatch.setattr(
        AimlapiLargeLanguageModel,
        "_update_credential",
        lambda self, model_name, credentials: None,
    )

    def fail(*args, **kwargs):
        raise InvokeError(
            "API request failed with status code 401: raw-secret-response"
        )

    monkeypatch.setattr(AimlapiLargeLanguageModel, "_generate", fail)

    with pytest.raises(InvokeAuthorizationError, match="invalid or has been revoked"):
        model._invoke(
            "openai/gpt-4o-mini",
            {"api_key": "secret-key"},
            [],
            {},
            stream=False,
        )


def test_invoke_maps_stream_error(monkeypatch: pytest.MonkeyPatch):
    model = object.__new__(AimlapiLargeLanguageModel)
    monkeypatch.setattr(
        AimlapiLargeLanguageModel,
        "_update_credential",
        lambda self, model_name, credentials: None,
    )

    def failing_stream():
        yield "first"
        raise InvokeError(
            "API request failed with status code 401: raw-secret-response"
        )

    monkeypatch.setattr(
        AimlapiLargeLanguageModel,
        "_generate",
        lambda *args, **kwargs: failing_stream(),
    )

    result = model._invoke(
        "openai/gpt-4o-mini",
        {"api_key": "secret-key"},
        [],
        {},
        stream=True,
    )
    assert next(result) == "first"
    with pytest.raises(InvokeAuthorizationError, match="invalid or has been revoked"):
        next(result)
