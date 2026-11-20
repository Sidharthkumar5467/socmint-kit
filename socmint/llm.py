"""Thin, logged wrapper around the Anthropic SDK for analysis steps.

- Optional: `get_llm(cfg, run_dir)` returns None when ANTHROPIC_API_KEY is unset, and every
  caller must have a non-LLM fallback (keyword labels, heuristic stance, `unverified`).
- Every prompt and response is appended to `<run_dir>/llm_log.jsonl` so a reviewer can
  check exactly what the model saw (docs/SPEC.md §6, misinformation).
- Structured output uses `client.messages.parse(output_format=<pydantic model>)`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from socmint.config import redact
from socmint.util.log import get_logger
from socmint.util.pan import dumps_masked
from socmint.util.timeutil import now_iso

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, api_key: str, model: str, run_dir: Path | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=120.0)
        self.model = model
        self.log_path = (run_dir / "llm_log.jsonl") if run_dir else None
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def name(self) -> str:
        return f"llm:{self.model}"

    def _log(self, purpose: str, system: str, prompt: str, output: str, error: str | None):
        if not self.log_path:
            return
        rec = {
            "at": now_iso(),
            "model": self.model,
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "output": output,
            "error": error,
        }
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            # prompts carry post texts (already masked); evidence snippets may not be
            f.write(dumps_masked(rec) + "\n")

    def parse(
        self,
        schema: type[T],
        prompt: str,
        *,
        system: str = "",
        purpose: str = "",
        max_tokens: int = 4096,
    ) -> T:
        """One structured call. Raises LLMError (redacted) on any API/refusal failure."""
        a = self._anthropic
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system or a.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                output_config={"effort": "low"},
            )
        except (a.APIStatusError, a.APIConnectionError) as e:
            msg = redact(f"{type(e).__name__}: {e}")
            self._log(purpose, system, prompt, "", msg)
            raise LLMError(msg) from e
        except ValidationError as e:
            # The SDK validates inside parse(); a reply cut off by max_tokens (thinking
            # tokens count toward it) arrives here as invalid JSON, before we see the
            # response. Seen live: stance batches truncated at max_tokens=2048.
            msg = redact(f"model output was not valid {schema.__name__} JSON (likely cut off "
                         f"by max_tokens={max_tokens}): {str(e).splitlines()[0]}")
            self._log(purpose, system, prompt, "", msg)
            raise LLMError(msg) from e
        usage = getattr(resp, "usage", None)
        # Callers share one LLM across threads (narrative labelling); += is not atomic.
        with self._lock:
            self.calls += 1
            if usage is not None:
                self.input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            self._log(purpose, system, prompt, "", f"no parsed output ({resp.stop_reason})")
            raise LLMError(f"model returned no structured output ({resp.stop_reason})")
        out = resp.parsed_output
        self._log(purpose, system, prompt, out.model_dump_json(), None)
        return out

    def usage(self) -> dict[str, float]:
        return {
            "llm_calls": self.calls,
            "llm_input_tokens": self.input_tokens,
            "llm_output_tokens": self.output_tokens,
        }


def get_llm(cfg: dict[str, str], run_dir: Path | None = None) -> LLM | None:
    key = (cfg.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    try:
        return LLM(key, cfg.get("ANALYSIS_MODEL") or "claude-haiku-5-5", run_dir)
    except ImportError:
        log.warning("anthropic package not installed; LLM steps disabled")
        return None
