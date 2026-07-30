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

Version 0.0.4 includes a generated snapshot of every model exposed as
`openai/chat-completions` by the aimlapi.com catalog. Models marked as hottest
appear first; the rest follow in stable model-ID order. Pricing is the base
inference tier reported by the catalog when the snapshot is generated.

Run `uv run python scripts/sync_models.py` before a future catalog release to
refresh the checked-in YAML files from
`GET https://api.aimlapi.com/models?include=capabilities,pricing`.

Embeddings, audio, video, and image generation remain out of scope.

## Optional device authorization

The provider help URL opts into Dify's proposed model-provider device
authorization hook with `dify_device_authorization=api_key`.

Supported Dify versions start the existing aimlapi.com Device Authorization
Grant server-side, open the standard aimlapi.com consent page, and poll for the
generated key. Dify then inserts the key into the declared secret field; the
user still confirms it through the normal Save action.

No custom aimlapi.com callback page is required. On Dify versions without the
hook, the same URL remains a regular key-management link and the key can be
pasted manually.

The v0.0.4 staging package requires the companion Dify core branch for browser
key return and the in-form low-balance checkout prompt:

https://github.com/aimlapi/dify/tree/d1m7asis/model-provider-oauth-callback

The current package intentionally points its onboarding link to the AIMLAPI
staging environment and must not be published to the production Marketplace.
