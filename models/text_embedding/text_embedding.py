import json
from urllib.parse import urljoin

from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model import AIModelEntity, EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.errors.model import InvokeError, InvokeServerUnavailableError
import requests

from provider.aimlapi_client import (
    inference_base_url,
    invoke_error_for,
    partner_headers,
    validate_api_key,
)


class AimlapiTextEmbeddingModel(OAICompatEmbeddingModel):
    """aimlapi.com text embedding models (OpenAI-compatible /embeddings).

    The OpenAI-compatible base builds its own request and does not forward the
    aimlapi.com attribution headers, so `_invoke` is reimplemented here to send
    `partner_headers()` on every request and to map upstream failures through the
    shared `invoke_error_for` (auth / balance / rate-limit / unavailable).
    """

    def _update_credential(self, credentials: dict) -> None:
        credentials["endpoint_url"] = inference_base_url()

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: str | None = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        del input_type
        self._update_credential(credentials)
        api_key = str(credentials.get("api_key") or "")

        headers = {"Content-Type": "application/json", **partner_headers()}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = inference_base_url()
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        endpoint_url = urljoin(endpoint_url, "embeddings")

        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)

        inputs: list[str] = []
        for text in texts:
            num_tokens = self._get_num_tokens_by_gpt2(text)
            if num_tokens >= context_size:
                cutoff = int((len(text) * context_size) // num_tokens)
                inputs.append(text[0:cutoff])
            else:
                inputs.append(text)

        batched_embeddings: list[list[float]] = []
        used_tokens = 0

        try:
            for i in range(0, len(inputs), max_chunks):
                payload: dict = {
                    "input": inputs[i : i + max_chunks],
                    "model": model,
                    "encoding_format": "float",
                }
                if user:
                    payload["user"] = user

                response = requests.post(
                    endpoint_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=(10, 300),
                )
                response.raise_for_status()
                response_data = response.json()

                batched_embeddings += [
                    item["embedding"] for item in response_data["data"]
                ]
                used_tokens += response_data.get("usage", {}).get("total_tokens", 0)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            raise invoke_error_for(
                InvokeError(f"status code {status}"), api_key
            ) from error
        except requests.RequestException as error:
            raise InvokeServerUnavailableError(
                "aimlapi.com is temporarily unavailable. Please try again later."
            ) from error

        usage = self._calc_response_usage(
            model=model, credentials=credentials, tokens=used_tokens
        )
        return TextEmbeddingResult(
            embeddings=batched_embeddings, usage=usage, model=model
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        validate_api_key(str(credentials.get("api_key") or ""))

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        self._update_credential(credentials)
        return super().get_customizable_model_schema(model, credentials)
