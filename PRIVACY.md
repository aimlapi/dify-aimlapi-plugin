# Privacy Policy

The aimlapi.com plugin sends model requests from Dify to aimlapi.com at
`https://api.aimlapi.com/v1`.

## Data sent to aimlapi.com

When you use this plugin, Dify sends the prompt, conversation messages, selected
model ID, model parameters, tool-call context when enabled, and your aimlapi.com
key to aimlapi.com so the selected model can generate a response.

## Data stored by the plugin

The plugin does not store prompts, generated responses, or API keys itself. API
keys are managed by Dify's credential storage.

## Third-party services

Requests are processed by aimlapi.com and, depending on the selected model, may be
routed to the underlying model provider through aimlapi.com. Review aimlapi.com's
privacy and data-processing terms before using the plugin with sensitive data.

## Logs

The plugin does not intentionally log request content or secrets. Runtime errors
may be surfaced by Dify for troubleshooting, but API keys should not be included
in plugin error messages.
