from http import HTTPStatus

import pytest
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
import requests

from provider import aimlapi_client


class FakeResponse:
    def __init__(self, status_code: int, payload: object | None = None):
        self.status_code = status_code
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> object:
        return self._payload


def test_partner_headers_include_required_attribution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AIMLAPI_PARTNER_ID", raising=False)

    assert aimlapi_client.partner_headers() == {
        "HTTP-Referer": "https://dify.ai/",
        "X-Title": "Dify",
        "X-AIMLAPI-Source": "agent/dify",
        "X-AIMLAPI-Partner-ID": "part_D4fY8kN2vR7mX5qL9cT3wBgH",
    }


def test_validate_api_key_accepts_zero_balance_response(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict = {}

    def fake_get(url: str, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse(HTTPStatus.OK, {"balance": 0})

    monkeypatch.setattr(aimlapi_client.requests, "get", fake_get)
    aimlapi_client.validate_api_key("secret-key")

    assert seen["url"] == "https://api.aimlapi.com/v2/billing"
    assert seen["headers"]["Authorization"] == "Bearer secret-key"
    assert seen["headers"]["X-AIMLAPI-Source"] == "agent/dify"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (HTTPStatus.UNAUTHORIZED, "invalid or has been revoked"),
        (HTTPStatus.FORBIDDEN, "could not validate this key"),
        (HTTPStatus.TOO_MANY_REQUESTS, "rate limited"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "temporarily unavailable"),
    ],
)
def test_validate_api_key_sanitizes_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: str
):
    monkeypatch.setattr(
        aimlapi_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(status, {"secret": "raw-body"}),
    )

    with pytest.raises(CredentialsValidateFailedError, match=expected) as caught:
        aimlapi_client.validate_api_key("secret-key")

    assert "raw-body" not in str(caught.value)
    assert "secret-key" not in str(caught.value)


def test_validate_api_key_sanitizes_timeout(monkeypatch: pytest.MonkeyPatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("raw timeout detail")

    monkeypatch.setattr(aimlapi_client.requests, "get", timeout)

    with pytest.raises(CredentialsValidateFailedError, match="try again") as caught:
        aimlapi_client.validate_api_key("secret-key")

    assert "raw timeout detail" not in str(caught.value)


def test_checkout_uses_partner_headers_and_by_key_topup(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        if url.endswith("/v3/partner-checkout/sessions"):
            return FakeResponse(HTTPStatus.OK, {"sessionToken": "session-token"})
        return FakeResponse(
            HTTPStatus.OK,
            {"checkout": {"payUrl": "https://checkout.stripe.com/c/pay/test"}},
        )

    monkeypatch.setattr(aimlapi_client.requests, "post", fake_post)
    checkout_url = aimlapi_client.create_balance_checkout_url("secret-key")

    assert checkout_url == "https://checkout.stripe.com/c/pay/test"
    assert (
        calls[0]["headers"]["X-AIMLAPI-Partner-ID"] == aimlapi_client.AIMLAPI_PARTNER_ID
    )
    assert "Authorization" not in calls[0]["headers"]
    assert calls[1]["headers"]["Authorization"] == "Bearer secret-key"
    assert calls[1]["json"]["amountUsdMinor"] == 2500
    assert calls[1]["json"]["autoTopUp"] is True
    assert calls[1]["json"]["paymentSessionId"]


@pytest.mark.parametrize(
    "pay_url",
    [
        "http://checkout.stripe.com/c/pay/test",
        "https://stripe.example/c/pay/test",
        "file:///tmp/checkout",
    ],
)
def test_checkout_rejects_untrusted_pay_urls(
    monkeypatch: pytest.MonkeyPatch, pay_url: str
):
    responses = iter(
        [
            FakeResponse(HTTPStatus.OK, {"sessionToken": "session-token"}),
            FakeResponse(HTTPStatus.OK, {"checkout": {"payUrl": pay_url}}),
        ]
    )
    monkeypatch.setattr(
        aimlapi_client.requests, "post", lambda *args, **kwargs: next(responses)
    )

    assert (
        aimlapi_client.create_balance_checkout_url("secret-key")
        == aimlapi_client.AIMLAPI_BILLING_URL
    )


def test_untrusted_endpoint_never_receives_headers(monkeypatch: pytest.MonkeyPatch):
    called = False

    def fake_get(*args, **kwargs):
        nonlocal called
        called = True
        return FakeResponse(HTTPStatus.OK)

    monkeypatch.setenv("AIMLAPI_INFERENCE_URL", "https://attacker.example/v1")
    monkeypatch.setattr(aimlapi_client.requests, "get", fake_get)

    with pytest.raises(CredentialsValidateFailedError):
        aimlapi_client.validate_api_key("secret-key")

    assert called is False


@pytest.mark.parametrize(
    ("status", "expected_type", "expected_text"),
    [
        (401, InvokeAuthorizationError, "invalid or has been revoked"),
        (429, InvokeRateLimitError, "rate limit reached"),
        (500, InvokeServerUnavailableError, "temporarily unavailable"),
    ],
)
def test_runtime_errors_are_typed_and_sanitized(
    status: int, expected_type: type[InvokeError], expected_text: str
):
    original = InvokeError(
        f"API request failed with status code {status}: raw-secret-response"
    )

    mapped = aimlapi_client.invoke_error_for(original, "secret-key")

    assert isinstance(mapped, expected_type)
    assert expected_text in str(mapped)
    assert "raw-secret-response" not in str(mapped)
    assert "secret-key" not in str(mapped)


def test_runtime_403_returns_checkout_markdown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        aimlapi_client,
        "create_balance_checkout_url",
        lambda api_key: "https://checkout.stripe.com/c/pay/test",
    )

    mapped = aimlapi_client.invoke_error_for(
        InvokeError("API request failed with status code 403: hidden"),
        "secret-key",
    )

    assert isinstance(mapped, InvokeBadRequestError)
    assert (
        str(mapped) == "aimlapi.com balance is insufficient. "
        "[Add $25 credits](https://checkout.stripe.com/c/pay/test)."
    )
