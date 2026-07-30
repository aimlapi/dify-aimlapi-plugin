from pathlib import Path

import yaml

from scripts import sync_models


def test_select_models_keeps_every_chat_model_and_ranks_hottest_first():
    entries = [
        {
            "id": "vendor/regular-b",
            "type": sync_models.CHAT_COMPLETIONS_TYPE,
            "info": {},
        },
        {
            "id": "openai/gpt-5.6-luna-pro",
            "type": sync_models.CHAT_COMPLETIONS_TYPE,
            "info": {"isHottest": True},
        },
        {
            "id": "vendor/hottest-extra",
            "type": sync_models.CHAT_COMPLETIONS_TYPE,
            "info": {"isHottest": True},
        },
        {
            "id": "vendor/regular-a",
            "type": sync_models.CHAT_COMPLETIONS_TYPE,
            "info": {},
        },
        {
            "id": "vendor/image",
            "type": "openai/images-generations",
            "info": {"isHottest": True},
        },
    ]

    selected = sync_models.select_models(entries)

    assert [entry["id"] for entry in selected] == [
        "openai/gpt-5.6-luna-pro",
        "vendor/hottest-extra",
        "vendor/regular-a",
        "vendor/regular-b",
    ]


def test_fetch_catalog_sends_required_attribution_headers(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"data": []}'

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(sync_models.urllib.request, "urlopen", fake_urlopen)

    assert sync_models.fetch_catalog() == []
    assert seen["request"].get_header("X-aimlapi-source") == "agent/dify"
    assert (
        seen["request"].get_header("X-aimlapi-partner-id")
        == "part_D4fY8kN2vR7mX5qL9cT3wBgH"
    )


def test_checked_in_catalog_has_expected_models_in_order():
    model_dir = Path(__file__).resolve().parents[1] / "models" / "llm"
    position = yaml.safe_load((model_dir / "_position.yaml").read_text("utf-8"))

    assert len(position) >= 300
    assert len(set(position)) == len(position)
    assert position[: len(sync_models.PREFERRED_HOTTEST_ORDER)] == (
        sync_models.PREFERRED_HOTTEST_ORDER
    )


def test_checked_in_model_metadata_matches_catalog_contract():
    model_dir = Path(__file__).resolve().parents[1] / "models" / "llm"

    position = yaml.safe_load((model_dir / "_position.yaml").read_text("utf-8"))
    for model_id in position:
        path = model_dir / sync_models.model_filename(model_id)
        document = yaml.safe_load(path.read_text("utf-8"))
        assert document["model"] == model_id
        assert document["label"]["en_US"]
        assert document["model_properties"]["context_size"] > 0
        assert float(document["pricing"]["input"]) >= 0
        assert float(document["pricing"]["output"]) >= 0
        assert document["pricing"]["unit"] == "0.000001"


def test_generator_maps_capabilities_and_base_prices():
    entry = {
        "id": "vendor/model",
        "info": {
            "name": "Model",
            "contextLength": 100_000,
            "outputMax": 8_000,
        },
        "capabilities": [
            "file_input",
            "parallel_tool_calls",
            "streaming",
            "structured_output",
            "tools",
            "vision",
        ],
        "pricing": {
            "units": [
                {
                    "type": "charge",
                    "name": "token",
                    "content": "text",
                    "phase": "inference",
                    "author": "user",
                    "origin": "provided",
                    "price": 2,
                    "per": 1_000_000,
                },
                {
                    "type": "charge",
                    "name": "token",
                    "content": "text",
                    "phase": "inference",
                    "author": "model",
                    "origin": "generated",
                    "price": 10,
                    "per": 1_000_000,
                },
            ]
        },
    }

    document = yaml.safe_load(sync_models.model_document(entry))

    assert document["label"]["en_US"] == "Model"
    assert document["model_properties"]["context_size"] == 100_000
    assert document["features"] == [
        "agent-thought",
        "tool-call",
        "multi-tool-call",
        "stream-tool-call",
        "vision",
        "document",
    ]
    assert document["pricing"]["input"] == "2"
    assert document["pricing"]["output"] == "10"
