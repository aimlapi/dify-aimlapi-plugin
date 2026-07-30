import os
import re
from http import HTTPStatus
from urllib.parse import urlparse
from uuid import uuid4

from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
import requests

AIMLAPI_PARTNER_ID = "part_D4fY8kN2vR7mX5qL9cT3wBgH"
AIMLAPI_PARTNER_NAME = "dify"
AIMLAPI_SOURCE = "agent/dify"
AIMLAPI_BILLING_URL = "https://aimlapi.com/app/billing/"
AIMLAPI_CHECKOUT_AMOUNT_USD_MINOR = 2500
REQUEST_TIMEOUT_SECONDS = 15

_STATUS_CODE_RE = re.compile(r"\bstatus code (?P<status>\d{3})\b", re.IGNORECASE)


def _env_url(name: str, default: str) -> str:
    return os.getenv(name, default).strip().rstrip("/")


def inference_base_url() -> str:
    return _env_url("AIMLAPI_INFERENCE_URL", "https://api.aimlapi.com/v1")


def app_base_url() -> str:
    return _env_url("AIMLAPI_APP_URL", "https://app.aimlapi.com")


def pay_base_url() -> str:
    return _env_url("AIMLAPI_PAY_URL", "https://pay.aimlapi.com")


def partner_headers() -> dict[str, str]:
    partner_id = os.getenv("AIMLAPI_PARTNER_ID", AIMLAPI_PARTNER_ID).strip()
    return {
        "HTTP-Referer": "https://dify.ai/",
        "X-Title": "Dify",
        "X-AIMLAPI-Source": AIMLAPI_SOURCE,
        "X-AIMLAPI-Partner-ID": partner_id or AIMLAPI_PARTNER_ID,
    }


def _is_trusted_aimlapi_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "aimlapi.com" or hostname.endswith(".aimlapi.com")
    )


def is_trusted_checkout_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "checkout.stripe.com"
        or hostname == "aimlapi.com"
        or hostname.endswith(".aimlapi.com")
    )


def _trusted_headers(url: str, api_key: str | None = None) -> dict[str, str]:
    if not _is_trusted_aimlapi_url(url):
        raise ValueError(
            "Refusing to send aimlapi.com credentials to an untrusted host"
        )

    headers = {"Content-Type": "application/json", **partner_headers()}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def validate_api_key(api_key: str) -> None:
    key = api_key.strip()
    if not key:
        raise CredentialsValidateFailedError("API key is required")

    url = f"{inference_base_url().removesuffix('/v1')}/v2/billing"
    try:
        response = requests.get(
            url,
            headers=_trusted_headers(url, key),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        raise CredentialsValidateFailedError(
            "aimlapi.com credential validation is not configured correctly."
        ) from exc
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise CredentialsValidateFailedError(
            "aimlapi.com could not validate the key right now. Please try again."
        ) from exc
    except requests.RequestException as exc:
        raise CredentialsValidateFailedError(
            "aimlapi.com credential validation failed. Please try again."
        ) from exc

    if response.status_code == HTTPStatus.OK:
        return
    if response.status_code == HTTPStatus.UNAUTHORIZED:
        raise CredentialsValidateFailedError(
            "The aimlapi.com key is invalid or has been revoked."
        )
    if response.status_code == HTTPStatus.FORBIDDEN:
        raise CredentialsValidateFailedError(
            "aimlapi.com could not validate this key. Check its account access and try again."
        )
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        raise CredentialsValidateFailedError(
            "aimlapi.com credential validation is rate limited. Please try again shortly."
        )
    if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        raise CredentialsValidateFailedError(
            "aimlapi.com is temporarily unavailable. Please try again later."
        )
    raise CredentialsValidateFailedError(
        "aimlapi.com could not validate this key. Please try again."
    )


def _checkout_return_url(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    checkout = result.get("checkout")
    if not isinstance(checkout, dict):
        return None
    pay_url = checkout.get("payUrl")
    if not isinstance(pay_url, str) or not is_trusted_checkout_url(pay_url):
        return None
    return pay_url


def create_balance_checkout_url(api_key: str) -> str:
    session_url = f"{app_base_url()}/v3/partner-checkout/sessions"
    topup_url = f"{inference_base_url().removesuffix('/v1')}/v2/billing/topup"
    return_base = pay_base_url()

    try:
        if not _is_trusted_aimlapi_url(return_base):
            return AIMLAPI_BILLING_URL
        session_response = requests.post(
            session_url,
            headers=_trusted_headers(session_url),
            json={
                "partnerId": partner_headers()["X-AIMLAPI-Partner-ID"],
                "partnerName": AIMLAPI_PARTNER_NAME,
                "returnUrl": AIMLAPI_BILLING_URL,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not session_response.ok:
            return AIMLAPI_BILLING_URL

        session = session_response.json()
        session_token = (
            session.get("sessionToken") if isinstance(session, dict) else None
        )
        if not isinstance(session_token, str) or not session_token.strip():
            return AIMLAPI_BILLING_URL

        success_url = (
            f"{return_base}/checkout?checkout=success&partnerCheckout=1"
            f"&sessionToken={session_token}"
        )
        cancel_url = (
            f"{return_base}/checkout?checkout=cancel&partnerCheckout=1"
            f"&sessionToken={session_token}"
        )
        topup_response = requests.post(
            topup_url,
            headers=_trusted_headers(topup_url, api_key.strip()),
            json={
                "sessionToken": session_token,
                "amountUsdMinor": AIMLAPI_CHECKOUT_AMOUNT_USD_MINOR,
                "paymentSessionId": str(uuid4()),
                "autoTopUp": True,
                "successUrl": success_url,
                "cancelUrl": cancel_url,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not topup_response.ok:
            return AIMLAPI_BILLING_URL
        return _checkout_return_url(topup_response.json()) or AIMLAPI_BILLING_URL
    except (ValueError, requests.RequestException):
        return AIMLAPI_BILLING_URL


def invoke_error_for(error: InvokeError, api_key: str) -> InvokeError:
    match = _STATUS_CODE_RE.search(str(error))
    status = int(match.group("status")) if match else None

    if status == HTTPStatus.UNAUTHORIZED:
        return InvokeAuthorizationError(
            "The aimlapi.com key is invalid or has been revoked."
        )
    if status == HTTPStatus.FORBIDDEN:
        checkout_url = create_balance_checkout_url(api_key)
        return InvokeBadRequestError(
            f"aimlapi.com balance is insufficient. [Add $25 credits]({checkout_url})."
        )
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return InvokeRateLimitError(
            "aimlapi.com rate limit reached. Please try again shortly."
        )
    if status is not None and status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return InvokeServerUnavailableError(
            "aimlapi.com is temporarily unavailable. Please try again later."
        )
    return InvokeBadRequestError(
        "aimlapi.com request failed. Check the model ID and request parameters."
    )
