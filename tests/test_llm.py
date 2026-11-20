"""socmint.llm wrapper: failures become a logged, redacted LLMError (no network)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from socmint.llm import LLM, LLMError


class Items(BaseModel):
    items: list[str]


class _TruncatingMessages:
    """messages.parse that fails the way the SDK does on a reply cut off by max_tokens."""

    def parse(self, **_kw):
        Items.model_validate_json('{"items":["a","b')  # raises pydantic ValidationError


def test_truncated_structured_output_is_llmerror_and_logged(tmp_path):
    llm = LLM("sk-test", "claude-haiku-5-5", tmp_path)
    llm.client.messages = _TruncatingMessages()
    with pytest.raises(LLMError, match="cut off by max_tokens=123"):
        llm.parse(Items, "prompt", purpose="claims.stance:N4", max_tokens=123)
    rec = json.loads((tmp_path / "llm_log.jsonl").read_text().strip())
    assert rec["purpose"] == "claims.stance:N4"
    assert "not valid Items JSON" in rec["error"]
