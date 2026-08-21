# aimlapi.com

aimlapi.com provides OpenAI-compatible access to multiple chat model families through a single API key.

## Configuration

1. Open the `Get your API key from aimlapi.com` link or create a key at
   https://aimlapi.com/app/keys.
2. Add the aimlapi.com provider in Dify.
3. Paste your API key.
4. Select a predefined chat model or add a custom OpenAI-compatible model ID.

An account with a zero balance can still save its API key. If the balance is
insufficient during inference, the plugin returns a link to add $25 in credits.
You can also manage billing at https://aimlapi.com/app/billing/.

The provider uses this fixed endpoint:

```text
https://api.aimlapi.com/v1
```

## Included models

Version 0.0.9 includes a generated snapshot of every model exposed as
`openai/chat-completions` by the aimlapi.com catalog. Models marked as hottest
appear first; the rest follow in stable model-ID order. Pricing is the base
inference tier reported by the catalog when the snapshot is generated.

Run `uv run python scripts/sync_models.py` before a future catalog release to
refresh the checked-in YAML files from
`GET https://api.aimlapi.com/models?include=capabilities,pricing`.

Text embeddings are included as a separate model type. Audio, video, and
image generation remain out of scope.

## Optional device authorization

The provider help URL opts into Dify's proposed model-provider device
authorization hook with `dify_device_authorization=api_key`.

Supported Dify versions start the existing aimlapi.com Device Authorization
Grant server-side, open the standard aimlapi.com consent page, and poll for the
generated key. Dify then inserts the key into the declared secret field; the
user still confirms it through the normal Save action.

No custom aimlapi.com callback page is required.

`dify_device_authorization` is an aimlapi.com proposal that no released Dify
version implements yet, so on every Dify build available today the help URL
behaves as a plain key-management link and the key is pasted manually. Nothing
about the provider depends on the hook landing.

Browser key return and the in-form low-balance checkout prompt additionally
require the companion Dify core branch, which is not merged upstream:

https://github.com/aimlapi/dify/tree/d1m7asis/model-provider-oauth-callback

This package points at production aimlapi.com endpoints and is the one intended
for the Marketplace. To test against staging, override the endpoints at runtime
(`AIMLAPI_INFERENCE_URL`, `AIMLAPI_APP_URL`, `AIMLAPI_PAY_URL`) instead of
editing the checked-in URLs.
