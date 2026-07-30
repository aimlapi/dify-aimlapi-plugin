"""Generate the checked-in aimlapi.com Dify model catalog."""

import json
import sys
import urllib.request
from decimal import Decimal
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from provider.aimlapi_client import partner_headers  # noqa: E402

MODELS_URL = "https://api.aimlapi.com/models?include=capabilities,pricing"
CHAT_COMPLETIONS_TYPE = "openai/chat-completions"

PREFERRED_HOTTEST_ORDER = [
    "openai/gpt-5.6-luna-pro",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-sol-pro",
    "anthropic/claude-fable-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.6-flash",
    "google/gemini-3-6-flash",
    "zhipu/glm-5.2",
    "zhipu/glm-5-2",
    "alibaba/glm-5.2",
    "alibaba/qwen3.7-max",
    "minimax/minimax-m3",
    "moonshot/kimi-k3",
    "x-ai/grok-4-5",
]

FEATURE_ORDER = [
    "agent-thought",
    "tool-call",
    "multi-tool-call",
    "stream-tool-call",
    "vision",
    "document",
]


def fetch_catalog() -> list[dict]:
    request = urllib.request.Request(
        MODELS_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "dify-aimlapi-catalog/0.0.2",
            **partner_headers(),
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("aimlapi.com returned an unexpected model catalog shape")
    return entries


def select_models(entries: list[dict]) -> list[dict]:
    selected: dict[str, dict] = {}
    for entry in entries:
        if (
            isinstance(entry, dict)
            and entry.get("type") == CHAT_COMPLETIONS_TYPE
            and entry.get("id")
        ):
            model_id = str(entry["id"])
            if model_id in selected:
                raise ValueError(f"Duplicate chat-completions entry for {model_id}")
            selected[model_id] = entry

    if not selected:
        raise ValueError("No chat-completions models found in the aimlapi.com catalog")

    hottest = {
        model_id
        for model_id, entry in selected.items()
        if isinstance(entry.get("info"), dict)
        and entry["info"].get("isHottest") is True
    }
    ranked_hottest = [
        model_id for model_id in PREFERRED_HOTTEST_ORDER if model_id in hottest
    ]
    ranked_hottest.extend(sorted(hottest.difference(ranked_hottest)))
    remaining = sorted(selected.keys() - hottest)
    return [selected[model_id] for model_id in ranked_hottest + remaining]


def model_features(capabilities: list[str]) -> list[str]:
    source = set(capabilities)
    features: set[str] = set()
    if "tools" in source:
        features.update({"agent-thought", "tool-call"})
    if "parallel_tool_calls" in source:
        features.add("multi-tool-call")
    if "tools" in source and "streaming" in source:
        features.add("stream-tool-call")
    if "vision" in source:
        features.add("vision")
    if "file_input" in source:
        features.add("document")
    return [feature for feature in FEATURE_ORDER if feature in features]


def _token_price(entry: dict, *, author: str, origin: str) -> str:
    units = entry.get("pricing", {}).get("units", [])
    for unit in units:
        if (
            unit.get("type") == "charge"
            and unit.get("name") == "token"
            and unit.get("content") == "text"
            and unit.get("phase") == "inference"
            and unit.get("author") == author
            and unit.get("origin") == origin
        ):
            per = Decimal(str(unit.get("per") or 1))
            per_million = Decimal(str(unit["price"])) * Decimal(1_000_000) / per
            return format(per_million.normalize(), "f")
    raise ValueError(
        f"Missing {author}/{origin} base inference pricing for {entry.get('id')}"
    )


def model_document(entry: dict) -> str:
    info = entry.get("info") or {}
    model_id = str(entry["id"])
    label = str(info.get("name") or model_id)
    context_size = int(info.get("contextLength") or 4096)
    output_max = int(info.get("outputMax") or min(context_size, 16384))
    default_max = min(output_max, 4096)
    features = model_features(list(entry.get("capabilities") or []))

    lines = [
        f"model: {json.dumps(model_id)}",
        "label:",
        f"  en_US: {json.dumps(label)}",
        "model_type: llm",
    ]
    if features:
        lines.append("features:")
        lines.extend(f"  - {feature}" for feature in features)
    lines.extend(
        [
            "model_properties:",
            "  mode: chat",
            f"  context_size: {context_size}",
            "parameter_rules:",
            "  - name: temperature",
            "    use_template: temperature",
            "  - name: top_p",
            "    use_template: top_p",
            "  - name: presence_penalty",
            "    use_template: presence_penalty",
            "  - name: frequency_penalty",
            "    use_template: frequency_penalty",
            "  - name: max_tokens",
            "    use_template: max_tokens",
            f"    default: {default_max}",
            "    min: 1",
            f"    max: {output_max}",
        ]
    )
    if "structured_output" in set(entry.get("capabilities") or []):
        lines.extend(
            [
                "  - name: response_format",
                "    use_template: response_format",
            ]
        )
    lines.extend(
        [
            "pricing:",
            f'  input: "{_token_price(entry, author="user", origin="provided")}"',
            f'  output: "{_token_price(entry, author="model", origin="generated")}"',
            '  unit: "0.000001"',
            "  currency: USD",
            "",
        ]
    )
    return "\n".join(lines)


def model_filename(model_id: str) -> str:
    return f"{model_id.replace('/', '-')}.yaml"


def write_catalog(entries: list[dict], destination: Path) -> None:
    selected = select_models(entries)
    destination.mkdir(parents=True, exist_ok=True)

    model_ids = [str(entry["id"]) for entry in selected]
    expected_files = {model_filename(model_id) for model_id in model_ids}
    for path in destination.glob("*.yaml"):
        if path.name != "_position.yaml" and path.name not in expected_files:
            path.unlink()

    for entry in selected:
        (destination / model_filename(str(entry["id"]))).write_text(
            model_document(entry), encoding="utf-8", newline="\n"
        )
    (destination / "_position.yaml").write_text(
        "".join(f"- {model_id}\n" for model_id in model_ids),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "models" / "llm"
    entries = fetch_catalog()
    write_catalog(entries, destination)
    print(
        f"Generated {len(select_models(entries))} "
        f"aimlapi.com model definitions in {destination}"
    )


if __name__ == "__main__":
    main()
