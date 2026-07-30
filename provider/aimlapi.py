from dify_plugin import ModelProvider

from provider.aimlapi_client import validate_api_key


class AimlapiProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        validate_api_key(str((credentials or {}).get("api_key") or ""))
