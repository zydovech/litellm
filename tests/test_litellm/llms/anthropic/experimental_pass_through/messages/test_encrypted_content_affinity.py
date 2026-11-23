"""
Tests for `/v1/messages` encrypted-content-aware deployment affinity.

Covers:
- Hardened `litellm_enc:` codec round-tripping and malformed-input rejection
- Non-streaming wrapping & assistant-only reverse deployment ID extraction
- Scope isolation (`tool_use.input`, `server_tool_use.input`, media `source`,
  `user` / `tool_result` / `system` messages never wrapped, unwrapped, or used for routing)
- Request preparation (same deployment keep, per-block mismatched thinking strip
  preserving earlier matching turns & prompt cache, empty-message removal,
  cross-deployment foreign `signature`/`thought_signature` stripping on `tool_use`,
  `encrypted_*` preservation, immutability across Router retries/fallbacks)
- Streaming SSE wrapping (multi-chunk `signature_delta`, `redacted_thinking`,
  `citations_delta`, `web_search_tool_result`, arbitrary TCP/UTF-8 splits,
  mixed `\r\n`/`\n` line endings, `aclose()` forwarding, fast-path bypass,
  multi-phase agentic streams)
- Handler integration (async native, async adapter, OpenAI Responses adapter
  double-wrap isolation, sync, agentic recursion guard)
- Router end-to-end flows (pinning, cooldown failover, retry, fallback, chat regression)
- High-concurrency isolation and fast-path performance benchmarks
"""

import asyncio
import base64
import copy
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.encrypted_content_affinity import (
    AnthropicEncryptedContentAffinityStreamIterator,
    apply_anthropic_response_affinity_wrapping,
    extract_latest_model_id_from_anthropic_messages,
    get_anthropic_affinity_context,
    prepare_anthropic_messages_for_deployment,
    wrap_anthropic_messages_response,
)
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages,
    anthropic_messages_handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
    EncryptedContentAffinityCheck,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

# ---------------------------------------------------------------------------
# Shared test helpers (reused across unit, router, concurrency, perf tests)
# ---------------------------------------------------------------------------


def _wrap(value: str, model_id: str) -> str:
    return ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
        value, model_id
    )


def _unwrap(value: str) -> tuple[Optional[str], str]:
    return ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(value)


def _sse_event(event_type: str, payload: Dict[str, Any], sep: str = "\n\n") -> bytes:
    line_end = "\r\n" if "\r\n" in sep else "\n"
    return (
        f"event: {event_type}{line_end}"
        f"data: {json.dumps(payload, ensure_ascii=False)}{sep}"
    ).encode("utf-8")


async def _iter_chunks(
    chunks: List[Any], yield_between: bool = False
) -> AsyncIterator[Any]:
    for chunk in chunks:
        if yield_between:
            await asyncio.sleep(0)
        yield chunk


async def _collect_sse_events(stream: AsyncIterator[Any]) -> List[Dict[str, Any]]:
    raw_bytes = bytearray()
    async for chunk in stream:
        if isinstance(chunk, (bytes, bytearray)):
            raw_bytes.extend(chunk)
        elif isinstance(chunk, str):
            raw_bytes.extend(chunk.encode("utf-8"))
    text = raw_bytes.decode("utf-8")
    events: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def _make_rich_response(
    sig: str = "sig_abc_123",
    redacted: str = "redacted_blob_456",
    web_enc: str = "web_enc_789",
    citation_idx: str = "cite_idx_012",
    stdout_enc: str = "stdout_enc_345",
    compaction_enc: str = "compact_enc_678",
    tool_sig: str = "gemini_thought_sig_901",
    image_data: str = "iVBORw0KGgoAAAANSUhEUg==",
) -> Dict[str, Any]:
    return {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [
            {"type": "thinking", "thinking": "Step 1", "signature": sig},
            {"type": "thinking", "thinking": "Empty sig", "signature": ""},
            {"type": "redacted_thinking", "data": redacted},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Docs",
                        "url": "https://example.com",
                        "encrypted_content": web_enc,
                    }
                ],
            },
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {
                    "type": "encrypted_code_execution_result",
                    "encrypted_stdout": stdout_enc,
                    "stderr": "",
                    "return_code": 0,
                    "content": [],
                },
            },
            {"type": "compaction", "encrypted_content": compaction_enc},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
                "provider_specific_fields": {
                    "signature": tool_sig,
                    "thought_signature": f"thought_{tool_sig}",
                },
            },
            {
                "type": "text",
                "text": "Sunny in Tokyo",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "cited_text": "Tokyo weather",
                        "url": "https://example.com",
                        "title": "Docs",
                        "encrypted_index": citation_idx,
                    }
                ],
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _make_anthropic_router(
    deployment_ids: List[str],
    model_name: str = "claude-sonnet-4-6",
    num_retries: int = 0,
    allowed_fails: int = 0,
    fallbacks: Optional[List[Dict[str, List[str]]]] = None,
    extra_model_list: Optional[List[Dict[str, Any]]] = None,
) -> litellm.Router:
    model_list: List[Dict[str, Any]] = [
        {
            "model_name": model_name,
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-6",
                "api_key": f"mock-key-{dep_id}",
            },
            "model_info": {"id": dep_id},
        }
        for dep_id in deployment_ids
    ]
    if extra_model_list:
        model_list.extend(extra_model_list)
    return litellm.Router(
        model_list=model_list,
        optional_pre_call_checks=["encrypted_content_affinity"],
        num_retries=num_retries,
        allowed_fails=allowed_fails,
        fallbacks=fallbacks or [],
    )


# ---------------------------------------------------------------------------
# 1. Hardened codec tests
# ---------------------------------------------------------------------------


class TestCodecHardening:
    @pytest.mark.parametrize(
        "malformed",
        [
            "plain_signature",
            "litellm_enc:",
            "litellm_enc:;payload",
            "litellm_enc:no_semicolon",
            "litellm_enc:!!!invalid_b64!!!;payload",
            # Valid base64 for "not_model_id:abc" — must NOT match
            f"litellm_enc:{base64.b64encode(b'other_key:abc').decode()};payload",
            # Valid base64 for "model_id:" with empty ID — must NOT match
            f"litellm_enc:{base64.b64encode(b'model_id:').decode()};payload",
        ],
    )
    def test_unwrap_rejects_malformed_or_non_matching_headers(self, malformed: str):
        model_id, content = _unwrap(malformed)
        assert model_id is None
        assert content == malformed

    def test_roundtrip_when_original_already_starts_with_litellm_enc(self):
        inner = _wrap("deep_secret", "upstream-inner")
        outer = _wrap(inner, "deploy-A")
        extracted_id, peeled = _unwrap(outer)
        assert extracted_id == "deploy-A"
        assert peeled == inner

    def test_unwrap_restores_stripped_base64_padding(self):
        wrapped = _wrap("sig_val", "dep-1")
        header, rest = wrapped.split(";", 1)
        unpadded = f"{header.rstrip('=')};{rest}"
        extracted_id, content = _unwrap(unpadded)
        assert extracted_id == "dep-1"
        assert content == "sig_val"


# ---------------------------------------------------------------------------
# 2. Non-streaming wrapping & reverse extraction
# ---------------------------------------------------------------------------


class TestExtractionAndWrapping:
    def test_wrap_response_wraps_all_encrypted_fields_and_preserves_media_data(self):
        original = _make_rich_response()
        snapshot = copy.deepcopy(original)

        wrapped = wrap_anthropic_messages_response(original, "deploy-A")

        # Input response dict must NOT be mutated in-place
        assert original == snapshot

        blocks = wrapped["content"]
        # thinking.signature wrapped; empty signature untouched
        assert _unwrap(blocks[0]["signature"]) == ("deploy-A", "sig_abc_123")
        assert blocks[1]["signature"] == ""
        # redacted_thinking.data wrapped
        assert _unwrap(blocks[2]["data"]) == ("deploy-A", "redacted_blob_456")
        # web_search_result.encrypted_content wrapped
        assert _unwrap(blocks[3]["content"][0]["encrypted_content"]) == (
            "deploy-A",
            "web_enc_789",
        )
        # encrypted_code_execution_result.encrypted_stdout wrapped
        assert _unwrap(blocks[4]["content"]["encrypted_stdout"]) == (
            "deploy-A",
            "stdout_enc_345",
        )
        # compaction.encrypted_content wrapped
        assert _unwrap(blocks[5]["encrypted_content"]) == (
            "deploy-A",
            "compact_enc_678",
        )
        # tool_use.provider_specific_fields.signature & thought_signature wrapped
        psf = blocks[6]["provider_specific_fields"]
        assert _unwrap(psf["signature"]) == ("deploy-A", "gemini_thought_sig_901")
        assert _unwrap(psf["thought_signature"]) == (
            "deploy-A",
            "thought_gemini_thought_sig_901",
        )
        # citations[].encrypted_index wrapped
        assert _unwrap(blocks[7]["citations"][0]["encrypted_index"]) == (
            "deploy-A",
            "cite_idx_012",
        )
        # image.source.data MUST NOT be wrapped
        assert blocks[8]["source"]["data"] == "iVBORw0KGgoAAAANSUhEUg=="

    def test_wrap_response_noop_on_missing_model_id_or_non_dict(self):
        resp = _make_rich_response()
        assert wrap_anthropic_messages_response(resp, None) is resp
        assert wrap_anthropic_messages_response(resp, "") is resp
        assert (
            wrap_anthropic_messages_response("not_a_dict", "deploy-A") == "not_a_dict"
        )
        no_content = {"id": "msg_empty", "type": "message"}
        assert wrap_anthropic_messages_response(no_content, "deploy-A") is no_content
        plain_resp = {
            "id": "msg_plain",
            "content": [{"type": "text", "text": "hello"}],
            # Top-level non-content fields must never be wrapped
            "metadata": {"signature": "outer_sig", "encrypted_flag": "outer_enc"},
        }
        assert wrap_anthropic_messages_response(plain_resp, "deploy-A") is plain_resp

    def test_wrap_response_never_modifies_tool_use_input_or_media_source(self):
        resp = {
            "id": "msg_tool_input",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_webhook",
                    "name": "verify_webhook",
                    "input": {
                        "signature": "hmac_sha256_user_payload",
                        "thought_signature": "user_field",
                        "encrypted_token": "user_secret_blob",
                        "nested": {"signature": "inner_user_sig"},
                    },
                    "provider_specific_fields": {"thought_signature": "gem_sig_1"},
                },
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "signature", "encrypted_filter": "raw_val"},
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "doc_bytes",
                        "signature": "doc_sig",
                    },
                },
            ],
        }
        wrapped = wrap_anthropic_messages_response(resp, "deploy-A")

        tool_block = wrapped["content"][0]
        # Tool arguments MUST remain bit-for-bit identical
        assert tool_block["input"] == resp["content"][0]["input"]
        # Provider signature on tool_use IS wrapped
        assert _unwrap(tool_block["provider_specific_fields"]["thought_signature"]) == (
            "deploy-A",
            "gem_sig_1",
        )
        # server_tool_use.input and document.source remain untouched
        assert wrapped["content"][1]["input"] == resp["content"][1]["input"]
        assert wrapped["content"][2]["source"] == resp["content"][2]["source"]

    def test_extract_latest_model_id_prefers_most_recent_turn_and_field(self):
        messages = [
            {"role": "user", "content": "turn 1"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t1",
                        "signature": _wrap("s1", "dep-A"),
                    },
                    {"type": "text", "text": "a1"},
                ],
            },
            {"role": "user", "content": "turn 2"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "a2",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "encrypted_index": _wrap("idx2", "dep-B"),
                            }
                        ],
                    }
                ],
            },
            # Trailing user tool_result with wrapped signature must NOT hijack routing
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "ok",
                                "signature": _wrap("hijack", "dep-HIJACK"),
                            }
                        ],
                    }
                ],
            },
        ]
        assert extract_latest_model_id_from_anthropic_messages(messages) == "dep-B"

    def test_extract_ignores_user_system_tool_input_and_media_data_fields(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "sys", "signature": _wrap("s", "dep-sys")}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _wrap("fake", "dep-fake")},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "encrypted_content": _wrap("hijack", "dep-user"),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _wrap("img", "dep-img"),
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "plain string content"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "verify",
                        "input": {
                            "signature": _wrap("arg_sig", "dep-arg"),
                            "encrypted_blob": _wrap("arg_enc", "dep-arg"),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "data": _wrap("img", "dep-img"),
                            "signature": _wrap("img_sig", "dep-img"),
                        },
                    },
                ],
            },
        ]
        assert extract_latest_model_id_from_anthropic_messages(messages) is None
        assert extract_latest_model_id_from_anthropic_messages([]) is None
        assert extract_latest_model_id_from_anthropic_messages("not-a-list") is None


# ---------------------------------------------------------------------------
# 3. Request preparation (keep vs strip, anti-hole, non-mutation)
# ---------------------------------------------------------------------------


class TestRequestPreparation:
    def test_fast_path_returns_same_list_when_no_wrapped_encrypted_fields(self):
        messages = [
            {"role": "user", "content": f"Literal mention of {_wrap('x', 'dep-1')}"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t", "signature": "raw_sig"}
                ],
            },
        ]
        prepared = prepare_anthropic_messages_for_deployment(messages, "dep-2")
        assert prepared is messages

    def test_same_deployment_unwraps_all_fields_without_mutating_input(self):
        rich = wrap_anthropic_messages_response(_make_rich_response(), "deploy-A")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": rich["content"]},
            {"role": "user", "content": "next"},
        ]
        original_snapshot = copy.deepcopy(messages)

        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-A")

        # Original messages untouched
        assert messages == original_snapshot
        # All fields restored to exact original values
        expected_content = _make_rich_response()["content"]
        assert prepared[1]["content"] == expected_content

    def test_mismatched_deployment_strips_thinking_and_foreign_tool_signatures(
        self,
    ):
        rich = wrap_anthropic_messages_response(_make_rich_response(), "deploy-A")
        messages = [
            {"role": "user", "content": "turn 1"},
            # Thinking-only assistant message -> must be omitted once thinking is stripped
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "only thought",
                        "signature": _wrap("s0", "deploy-A"),
                    }
                ],
            },
            # Mixed assistant message -> wrapped deploy-A thinking/redacted_thinking stripped,
            # unsigned empty-sig thinking preserved, foreign tool_use signatures stripped,
            # non-signature encrypted_* fields kept unwrapped.
            {"role": "assistant", "content": rich["content"]},
            {"role": "user", "content": "turn 2"},
        ]
        original_snapshot = copy.deepcopy(messages)

        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-B")

        assert messages == original_snapshot
        # Empty thinking-only message was omitted
        assert len(prepared) == 3
        kept_blocks = prepared[1]["content"]
        kept_types = [b["type"] for b in kept_blocks]
        # All deploy-A thinking & redacted_thinking blocks (including empty-sig) stripped
        assert "thinking" not in kept_types
        assert "redacted_thinking" not in kept_types
        # Non-thinking encrypted_* fields are unwrapped and preserved
        web_block = next(
            b for b in kept_blocks if b["type"] == "web_search_tool_result"
        )
        assert web_block["content"][0]["encrypted_content"] == "web_enc_789"
        code_block = next(
            b for b in kept_blocks if b["type"] == "code_execution_tool_result"
        )
        assert code_block["content"]["encrypted_stdout"] == "stdout_enc_345"
        compaction_block = next(b for b in kept_blocks if b["type"] == "compaction")
        assert compaction_block["encrypted_content"] == "compact_enc_678"
        # Foreign tool_use signatures stripped; empty provider_specific_fields dropped cleanly
        tool_block = next(b for b in kept_blocks if b["type"] == "tool_use")
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "Tokyo"}
        assert "provider_specific_fields" not in tool_block
        text_block = next(b for b in kept_blocks if b["type"] == "text")
        assert text_block["citations"][0]["encrypted_index"] == "cite_idx_012"
        image_block = next(b for b in kept_blocks if b["type"] == "image")
        assert image_block["source"]["data"] == "iVBORw0KGgoAAAANSUhEUg=="

    def test_cross_deployment_tool_signature_partial_cleanup_and_payload_isolation(
        self,
    ):
        literal_wrapped = _wrap("literal_user_payload", "deploy-A")
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "verify",
                        "input": {
                            "signature": literal_wrapped,
                            "thought_signature": literal_wrapped,
                            "encrypted_secret": literal_wrapped,
                        },
                        "provider_specific_fields": {
                            "signature": _wrap("gem_sig", "deploy-A"),
                            "thought_signature": _wrap("gem_tsig", "deploy-A"),
                            "keep_meta": "preserved",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "other",
                        "input": {"x": 1},
                        "provider_specific_fields": {
                            "signature": _wrap("sig_B", "deploy-B"),
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "done",
                        "signature": literal_wrapped,
                        "encrypted_content": literal_wrapped,
                    }
                ],
            },
        ]
        snapshot = copy.deepcopy(messages)

        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-B")
        assert messages == snapshot

        # toolu_1: input untouched; foreign signatures removed; non-signature metadata preserved
        t1 = prepared[0]["content"][0]
        assert t1["input"] == snapshot[0]["content"][0]["input"]
        assert t1["provider_specific_fields"] == {"keep_meta": "preserved"}
        # toolu_2: matching deploy-B signature unwrapped and kept
        t2 = prepared[0]["content"][1]
        assert t2["provider_specific_fields"] == {"signature": "sig_B"}
        # user tool_result passed through completely untouched
        assert prepared[1] is messages[1]

        # Unknown current_model_id (None) unwraps tool signatures without stripping
        prepared_unknown = prepare_anthropic_messages_for_deployment(messages, None)
        assert prepared_unknown[0]["content"][0]["provider_specific_fields"] == {
            "signature": "gem_sig",
            "thought_signature": "gem_tsig",
            "keep_meta": "preserved",
        }
        assert (
            prepared_unknown[0]["content"][0]["input"]
            == snapshot[0]["content"][0]["input"]
        )

    def test_mixed_history_a_then_b_keeps_b_and_strips_a(self):
        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "tA",
                        "signature": _wrap("sigA", "deploy-A"),
                    },
                    {"type": "text", "text": "a1"},
                ],
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "tB",
                        "signature": _wrap("sigB", "deploy-B"),
                    },
                    {"type": "text", "text": "a2"},
                ],
            },
        ]
        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-B")
        assert [b["type"] for b in prepared[1]["content"]] == ["text"]
        assert prepared[3]["content"] == [
            {"type": "thinking", "thinking": "tB", "signature": "sigB"},
            {"type": "text", "text": "a2"},
        ]

    def test_mixed_history_b_a_b_and_b_a_b_b_preserves_matching_turns(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t1",
                        "signature": _wrap("s1", "deploy-B"),
                    },
                    {"type": "text", "text": "a1"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": _wrap("r2", "deploy-A")},
                    {"type": "text", "text": "a2"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t3",
                        "signature": _wrap("s3", "deploy-B"),
                    },
                    {"type": "text", "text": "a3"},
                ],
            },
        ]
        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-B")
        # Turn 1 (B) & Turn 3 (B) kept & unwrapped (preserving B's prompt cache);
        # ONLY Turn 2 (A) stripped.
        assert prepared[0]["content"] == [
            {"type": "thinking", "thinking": "t1", "signature": "s1"},
            {"type": "text", "text": "a1"},
        ]
        assert prepared[1]["content"] == [{"type": "text", "text": "a2"}]
        assert prepared[2]["content"] == [
            {"type": "thinking", "thinking": "t3", "signature": "s3"},
            {"type": "text", "text": "a3"},
        ]

        # B -> A -> B -> B keeps all three B turns and strips only A
        messages_4 = messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t4",
                        "signature": _wrap("s4", "deploy-B"),
                    },
                    {"type": "text", "text": "a4"},
                ],
            }
        ]
        prepared_4 = prepare_anthropic_messages_for_deployment(messages_4, "deploy-B")
        assert [b["type"] for b in prepared_4[0]["content"]] == ["thinking", "text"]
        assert [b["type"] for b in prepared_4[1]["content"]] == ["text"]
        assert [b["type"] for b in prepared_4[2]["content"]] == ["thinking", "text"]
        assert prepared_4[3]["content"][0]["signature"] == "s4"

    def test_unknown_current_model_id_unwraps_without_stripping_thinking(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t1",
                        "signature": _wrap("s1", "deploy-A"),
                    },
                    {"type": "text", "text": "a1"},
                ],
            }
        ]
        prepared = prepare_anthropic_messages_for_deployment(messages, None)
        assert prepared[0]["content"][0]["signature"] == "s1"

    def test_handles_non_dict_messages_non_list_content_and_empty_list(self):
        assert prepare_anthropic_messages_for_deployment([], "deploy-A") == []
        user_wrapped = _wrap("e_user", "deploy-A")
        messages: List[Any] = [
            "raw_entry",
            {
                "role": "user",
                "content": ["raw_block", {"type": "text", "text": "ok"}],
                "extra": {"encrypted_content": user_wrapped},
            },
            {
                "role": "assistant",
                "content": [
                    "raw_assistant_block",
                    {
                        "type": "thinking",
                        "thinking": "bad header",
                        "signature": "litellm_enc:!!!bad!!!;keep_me",
                    },
                    {
                        "type": "compaction",
                        "encrypted_content": _wrap("e_assist", "deploy-A"),
                    },
                    {"type": "text", "text": "still here"},
                ],
            },
            {"role": "assistant", "content": []},
            {"role": "assistant", "content": "plain"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "unsigned", "signature": ""},
                    {"type": "text", "text": "keep unsigned"},
                ],
            },
        ]
        assert extract_latest_model_id_from_anthropic_messages(messages) == "deploy-A"
        prepared = prepare_anthropic_messages_for_deployment(messages, "deploy-B")
        assert prepared[0] == "raw_entry"
        # User message is untouched (never unwrapped)
        assert prepared[1] is messages[1]
        assert prepared[1]["extra"]["encrypted_content"] == user_wrapped
        # Assistant non-dict block & malformed litellm_enc header preserved; compaction unwrapped
        assert prepared[2]["content"][0] == "raw_assistant_block"
        assert prepared[2]["content"][1]["signature"] == "litellm_enc:!!!bad!!!;keep_me"
        assert prepared[2]["content"][2]["encrypted_content"] == "e_assist"
        # Originally empty / string assistant content is untouched and NOT omitted
        assert prepared[3] is messages[3]
        assert prepared[4] is messages[4]
        # Unsigned thinking block in an assistant turn without foreign markers is preserved
        assert prepared[5] is messages[5]


# ---------------------------------------------------------------------------
# 4. Streaming SSE wrapping tests
# ---------------------------------------------------------------------------


class TestStreamingSSEAffinity:
    @pytest.mark.asyncio
    async def test_streaming_wraps_multichunk_signature_redacted_citations_and_encrypted_blocks(
        self,
    ):
        raw_events = [
            _sse_event(
                "message_start",
                {"type": "message_start", "message": {"id": "msg_1", "content": []}},
            ),
            # Block 0: thinking with empty initial signature + empty delta + 2 non-empty signature deltas
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "思考中"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig_part_1_"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig_part_2"},
                },
            ),
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            # Block 1: redacted_thinking
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "redacted_thinking",
                        "data": "redacted_stream_blob",
                    },
                },
            ),
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 1}
            ),
            # Block 2: web_search_tool_result with encrypted_content
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "encrypted_content": "ws_enc_stream",
                            }
                        ],
                    },
                },
            ),
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 2}
            ),
            # Block 3: text with citations_delta carrying encrypted_index
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 3,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 3,
                    "delta": {"type": "text_delta", "text": "你好世界🌍"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 3,
                    "delta": {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "encrypted_index": "cite_stream_idx",
                        },
                    },
                },
            ),
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 3}
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ]

        # Fragment stream into tiny 7-byte TCP chunks splitting UTF-8 emojis & JSON boundaries
        full_wire = b"".join(raw_events)
        fragmented = [full_wire[i : i + 7] for i in range(0, len(full_wire), 7)]

        wrapped_stream = AnthropicEncryptedContentAffinityStreamIterator(
            stream=_iter_chunks(fragmented),
            model_id="deploy-stream-A",
        )
        events = await _collect_sse_events(wrapped_stream)

        # Client concatenates all signature_delta chunks for block 0
        sig_deltas = [
            e["delta"]["signature"]
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("index") == 0
            and e.get("delta", {}).get("type") == "signature_delta"
        ]
        reconstructed_sig = "".join(sig_deltas)
        assert _unwrap(reconstructed_sig) == (
            "deploy-stream-A",
            "sig_part_1_sig_part_2",
        )

        # Redacted thinking wrapped
        redacted_evt = next(
            e
            for e in events
            if e.get("type") == "content_block_start" and e.get("index") == 1
        )
        assert _unwrap(redacted_evt["content_block"]["data"]) == (
            "deploy-stream-A",
            "redacted_stream_blob",
        )

        # Web search encrypted_content wrapped
        ws_evt = next(
            e
            for e in events
            if e.get("type") == "content_block_start" and e.get("index") == 2
        )
        assert _unwrap(ws_evt["content_block"]["content"][0]["encrypted_content"]) == (
            "deploy-stream-A",
            "ws_enc_stream",
        )

        # Citations delta encrypted_index wrapped & multibyte UTF-8 text preserved
        text_evt = next(
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        )
        assert text_evt["delta"]["text"] == "你好世界🌍"
        cite_evt = next(
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "citations_delta"
        )
        assert _unwrap(cite_evt["delta"]["citation"]["encrypted_index"]) == (
            "deploy-stream-A",
            "cite_stream_idx",
        )

    @pytest.mark.asyncio
    async def test_streaming_initial_signature_on_start_prevents_double_wrap_and_resets_phase2(
        self,
    ):
        chunks = [
            # Phase 1: signature present directly in content_block_start + trailing delta
            _sse_event(
                "message_start",
                {"type": "message_start", "message": {"id": "msg_p1", "content": []}},
                sep="\r\n\r\n",
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "head_",
                    },
                },
                sep="\r\n\r\n",
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "tail"},
                },
                sep="\r\n\r\n",
            ),
            # Phase 2 starts abruptly without content_block_stop: message_start must clear index state
            _sse_event(
                "message_start",
                {"type": "message_start", "message": {"id": "msg_p2", "content": []}},
                sep="\r\n\r\n",
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                sep="\r\n\r\n",
            ),
            # Final frame intentionally missing trailing \r\n\r\n to test EOF flush
            b'event: content_block_delta\r\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "phase2_sig"}}',
        ]
        wrapped_stream = AnthropicEncryptedContentAffinityStreamIterator(
            stream=_iter_chunks(chunks), model_id="deploy-p"
        )
        events = await _collect_sse_events(wrapped_stream)

        # Phase 1 combined signature has exactly ONE prefix
        p1_start = events[1]["content_block"]["signature"]
        p1_delta = events[2]["delta"]["signature"]
        assert _unwrap(p1_start + p1_delta) == ("deploy-p", "head_tail")
        # Phase 2 signature was freshly wrapped
        p2_delta = events[5]["delta"]["signature"]
        assert _unwrap(p2_delta) == ("deploy-p", "phase2_sig")

    @pytest.mark.asyncio
    async def test_streaming_edge_cases_malformed_json_str_dict_and_non_dict_chunks(
        self,
    ):
        # Empty frames, comment mentioning 'signature', text_delta mentioning 'signature',
        # malformed JSON, non-dict JSON, and incomplete trailing UTF-8 byte sequence
        malformed_chunks = [
            b"\n\n: signature comment\n\n",
            b"event: content_block_delta\ndata: {not valid json with signature}\n\n",
            b'event: content_block_delta\ndata: ["signature"]\n\n',
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": "discussing signature field",
                    },
                },
            ),
            # Multi-line data: without space after colon + pre-populated message_start content
            (
                b'event: message_start\ndata:{"type": "message_start",\n'
                b'data:"message": {"id": "m", "content": [{"type": "thinking", "thinking": "x", "signature": "ms_sig"}]}}\n\n'
            ),
            # Incomplete trailing UTF-8 byte at EOF forces decoder tail flush
            b"\xe4",
        ]
        it = AnthropicEncryptedContentAffinityStreamIterator(
            _iter_chunks(malformed_chunks), "dep-1"
        )
        out_bytes = b"".join([c async for c in it])
        assert b"discussing signature field" in out_bytes
        assert b": signature comment" in out_bytes
        assert _wrap("ms_sig", "dep-1").encode("utf-8") in out_bytes
        # Calling __anext__ after exhaustion raises StopAsyncIteration cleanly
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

        # Split str chunks stay str
        str_chunks = [
            'event: content_block_delta\ndata: {"type": "content_block_delta", ',
            '"delta": {"type": "signature_delta", "signature": "s"}}\n\n',
        ]
        str_out = [
            c
            async for c in AnthropicEncryptedContentAffinityStreamIterator(
                _iter_chunks(str_chunks), "dep-str"
            )
        ]
        assert len(str_out) == 1
        assert isinstance(str_out[0], str)
        assert "litellm_enc:" in str_out[0]

        # dict chunks: message_start, malformed content_block_start, content_block_stop,
        # citations_delta (wrapped), non-content error event (untouched), and scalar passthrough
        dict_chunks: List[Any] = [
            {"type": "message_start", "message": {"id": "m_empty", "content": []}},
            {
                "type": "content_block_start",
                "index": "bad",
                "content_block": "not_a_dict",
            },
            {"type": "content_block_stop"},
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "citations_delta",
                    "citation": {"encrypted_index": "d_idx"},
                },
            },
            {
                "type": "error",
                "error": {"type": "api_error", "signature": "do_not_wrap"},
            },
            12345,
        ]
        dict_out = [
            c
            async for c in AnthropicEncryptedContentAffinityStreamIterator(
                _iter_chunks(dict_chunks), "dep-dict"
            )
        ]
        assert _unwrap(dict_out[3]["delta"]["citation"]["encrypted_index"]) == (
            "dep-dict",
            "d_idx",
        )
        assert dict_out[4]["error"]["signature"] == "do_not_wrap"
        assert dict_out[5] == 12345

        # Non-dict, non-iterator return value passes through untouched
        assert (
            apply_anthropic_response_affinity_wrapping(
                "raw_scalar",
                {
                    "litellm_metadata": {
                        "encrypted_content_affinity_enabled": True,
                        "model_info": {"id": "dep-1"},
                    }
                },
            )
            == "raw_scalar"
        )

    @pytest.mark.asyncio
    async def test_streaming_aclose_mixed_line_endings_and_tool_input_isolation(self):
        # 1. Mixed \r\n + \n in a single frame + tool_use.input & input_json_delta isolation
        mixed_chunks = [
            b'event: content_block_start\r\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "verify", "input": {"signature": "user_arg_sig"}, "provider_specific_fields": {"thought_signature": "gem_sig"}}}\n\n',
            b'event: content_block_delta\r\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"signature\\": \\"abc\\"}"}}\n\n',
            b'event: content_block_delta\r\ndata: {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "mixed_sig"}}\n\n',
        ]
        events = await _collect_sse_events(
            AnthropicEncryptedContentAffinityStreamIterator(
                _iter_chunks(mixed_chunks), "dep-mix"
            )
        )
        assert events[0]["content_block"]["input"] == {"signature": "user_arg_sig"}
        assert _unwrap(
            events[0]["content_block"]["provider_specific_fields"]["thought_signature"]
        ) == ("dep-mix", "gem_sig")
        assert events[1]["delta"]["partial_json"] == '{"signature": "abc"}'
        assert _unwrap(events[2]["delta"]["signature"]) == ("dep-mix", "mixed_sig")

        # 2. aclose() promptly closes inner async generator on early disconnect
        inner_closed = False

        async def _tracked_stream() -> AsyncIterator[bytes]:
            nonlocal inner_closed
            try:
                yield _sse_event(
                    "message_start",
                    {"type": "message_start", "message": {"id": "m1", "content": []}},
                )
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hi"},
                    },
                )
            finally:
                inner_closed = True

        wrapped = AnthropicEncryptedContentAffinityStreamIterator(
            _tracked_stream(), "dep-close"
        )
        first = await wrapped.__anext__()
        assert b"message_start" in first
        await wrapped.aclose()
        assert inner_closed is True
        with pytest.raises(StopAsyncIteration):
            await wrapped.__anext__()

        # 3. aclose() is safe even when inner async iterator has no aclose method
        class _MinimalAsyncIter:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        minimal_wrapped = AnthropicEncryptedContentAffinityStreamIterator(
            _MinimalAsyncIter(), "dep-min"
        )
        await minimal_wrapped.aclose()
        with pytest.raises(StopAsyncIteration):
            await minimal_wrapped.__anext__()


# ---------------------------------------------------------------------------
# 5. Messages handler integration (async native, adapter, sync, recursion guard)
# ---------------------------------------------------------------------------


class TestMessagesHandlerIntegration:
    @pytest.mark.asyncio
    async def test_async_handler_prepares_request_and_wraps_response(self):
        captured_messages: List[Any] = []

        async def fake_async_handler(*args, **kwargs):
            captured_messages.append(copy.deepcopy(kwargs["messages"]))
            return AnthropicMessagesResponse(**_make_rich_response())

        incoming = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "old",
                        "signature": _wrap("old_sig", "dep-A"),
                    },
                    {"type": "text", "text": "keep me"},
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        original = copy.deepcopy(incoming)

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            side_effect=fake_async_handler,
        ):
            resp = await anthropic_messages(
                max_tokens=100,
                messages=incoming,
                model="anthropic/claude-sonnet-4-6",
                thinking={"type": "adaptive"},
                api_key="test-key",
                litellm_metadata={
                    "encrypted_content_affinity_enabled": True,
                    "model_info": {"id": "dep-B"},
                },
            )

        # Caller messages untouched; upstream saw stripped thinking
        assert incoming == original
        assert [b["type"] for b in captured_messages[0][0]["content"]] == ["text"]
        # Response wrapped with dep-B
        assert _unwrap(resp["content"][0]["signature"]) == ("dep-B", "sig_abc_123")

    @pytest.mark.asyncio
    async def test_async_handler_agentic_loop_depth_prevents_double_wrap(self):
        resp = _make_rich_response()
        kwargs = {
            "_agentic_loop_depth": 1,
            "litellm_metadata": {
                "encrypted_content_affinity_enabled": True,
                "model_info": {"id": "dep-A"},
            },
        }
        result = apply_anthropic_response_affinity_wrapping(resp, kwargs)
        assert result is resp
        assert result["content"][0]["signature"] == "sig_abc_123"

    @pytest.mark.asyncio
    async def test_disabled_affinity_still_unwraps_stray_markers_without_wrapping_response(
        self,
    ):
        captured_messages: List[Any] = []

        async def fake_async_handler(*args, **kwargs):
            captured_messages.append(copy.deepcopy(kwargs["messages"]))
            return AnthropicMessagesResponse(**_make_rich_response())

        incoming = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t",
                        "signature": _wrap("orig_sig", "dep-A"),
                    },
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            side_effect=fake_async_handler,
        ):
            resp = await anthropic_messages(
                max_tokens=100,
                messages=incoming,
                model="anthropic/claude-sonnet-4-6",
                api_key="test-key",
                litellm_metadata={"model_info": {"id": "dep-A"}},
            )

        assert captured_messages[0][0]["content"][0]["signature"] == "orig_sig"
        # Not wrapped because encrypted_content_affinity_enabled is False
        assert resp["content"][0]["signature"] == "sig_abc_123"

    def test_sync_handler_and_adapter_path_wrap_and_context_fallback(self):
        # Context helper falls back to kwargs["model_info"] when litellm_metadata["model_info"] lacks id
        enabled, mid = get_anthropic_affinity_context(
            {
                "litellm_metadata": {
                    "encrypted_content_affinity_enabled": True,
                    "model_info": {},
                },
                "model_info": {"id": "fallback-id"},
            }
        )
        assert (enabled, mid) == (True, "fallback-id")

        captured_messages: List[Any] = []

        def fake_adapter_handler(**kwargs):
            captured_messages.append(copy.deepcopy(kwargs["messages"]))
            return AnthropicMessagesResponse(**_make_rich_response())

        incoming = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "t",
                        "signature": _wrap("s1", "dep-A"),
                    },
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
        with patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler",
            side_effect=fake_adapter_handler,
        ):
            resp = anthropic_messages_handler(
                max_tokens=100,
                messages=incoming,
                model="gemini/gemini-2.5-pro",
                api_key="test-key",
                litellm_metadata={
                    "encrypted_content_affinity_enabled": True,
                    "model_info": {"id": "dep-B"},
                },
            )

        assert [b["type"] for b in captured_messages[0][0]["content"]] == ["text"]
        assert _unwrap(resp["content"][0]["signature"]) == ("dep-B", "sig_abc_123")

    @pytest.mark.asyncio
    async def test_responses_api_adapter_disables_inner_affinity_without_double_wrap(
        self,
    ):
        captured_inner_metadata: List[Dict[str, Any]] = []

        async def fake_responses_adapter(**kwargs):
            captured_inner_metadata.append(copy.deepcopy(kwargs["litellm_metadata"]))
            return AnthropicMessagesResponse(**_make_rich_response())

        caller_metadata = {
            "encrypted_content_affinity_enabled": True,
            "model_info": {"id": "dep-OAI"},
        }
        with patch(
            "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler",
            side_effect=fake_responses_adapter,
        ):
            resp = await anthropic_messages(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-5",
                api_key="test-key",
                litellm_metadata=caller_metadata,
            )

        # Inner Responses API call saw affinity disabled to avoid double-wrapping
        assert captured_inner_metadata[0]["encrypted_content_affinity_enabled"] is False
        # Caller metadata dict was NOT mutated
        assert caller_metadata["encrypted_content_affinity_enabled"] is True
        # Outer /v1/messages wrapper wrapped the signature exactly once
        assert _unwrap(resp["content"][0]["signature"]) == ("dep-OAI", "sig_abc_123")
        assert not _unwrap(resp["content"][0]["signature"])[1].startswith(
            "litellm_enc:"
        )


# ---------------------------------------------------------------------------
# 6. Router end-to-end tests (pinning, cooldown, retry, fallback, regression)
# ---------------------------------------------------------------------------


class TestRouterAnthropicMessagesAffinity:
    @pytest.mark.asyncio
    async def test_router_pins_to_same_deployment_and_unwraps_thinking(self):
        router = _make_anthropic_router(["dep-1", "dep-2"])
        seen_calls: List[Dict[str, Any]] = []

        async def fake_upstream(*args, **kwargs):
            api_key = kwargs.get("api_key")
            seen_calls.append(
                {
                    "api_key": api_key,
                    "messages": copy.deepcopy(kwargs["messages"]),
                }
            )
            return AnthropicMessagesResponse(
                **_make_rich_response(sig=f"sig_from_{api_key}")
            )

        call_idx = 0

        def deterministic_choice(seq):
            nonlocal call_idx
            call_idx += 1
            return seq[0] if call_idx == 1 else seq[-1]

        with (
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
                side_effect=fake_upstream,
            ),
            patch(
                "litellm.router_strategy.simple_shuffle.random.choice",
                side_effect=deterministic_choice,
            ),
        ):
            r1 = await router.aanthropic_messages(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "turn 1"}],
            )
            assert seen_calls[0]["api_key"] == "mock-key-dep-1"
            assert _unwrap(r1["content"][0]["signature"]) == (
                "dep-1",
                "sig_from_mock-key-dep-1",
            )

            # Turn 2: normal shuffle would pick dep-2, but affinity pins back to dep-1
            r2 = await router.aanthropic_messages(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": "turn 1"},
                    {"role": "assistant", "content": r1["content"]},
                    {"role": "user", "content": "turn 2"},
                ],
            )
            assert seen_calls[1]["api_key"] == "mock-key-dep-1"
            # Thinking block kept and signature unwrapped cleanly
            assert seen_calls[1]["messages"][1]["content"][0]["type"] == "thinking"
            assert (
                seen_calls[1]["messages"][1]["content"][0]["signature"]
                == "sig_from_mock-key-dep-1"
            )
            assert _unwrap(r2["content"][0]["signature"]) == (
                "dep-1",
                "sig_from_mock-key-dep-1",
            )

    @pytest.mark.asyncio
    async def test_router_retry_switches_deployment_and_strips_mismatched_thinking(
        self,
    ):
        router = _make_anthropic_router(
            ["dep-1", "dep-2"], num_retries=1, allowed_fails=0
        )
        seen_calls: List[Dict[str, Any]] = []

        async def fake_upstream(*args, **kwargs):
            api_key = kwargs.get("api_key")
            seen_calls.append(
                {
                    "api_key": api_key,
                    "messages": copy.deepcopy(kwargs["messages"]),
                    "thinking": kwargs.get(
                        "anthropic_messages_optional_request_params", {}
                    ).get("thinking"),
                }
            )
            if api_key == "mock-key-dep-1":
                raise litellm.RateLimitError(
                    message="dep-1 overloaded",
                    llm_provider="anthropic",
                    model="claude-sonnet-4-6",
                )
            return AnthropicMessagesResponse(
                **_make_rich_response(sig="sig_from_dep_2")
            )

        rich_dep1 = wrap_anthropic_messages_response(_make_rich_response(), "dep-1")
        request_messages = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": rich_dep1["content"]},
            {"role": "user", "content": "turn 2"},
        ]

        with (
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
                side_effect=fake_upstream,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            response = await router.aanthropic_messages(
                model="claude-sonnet-4-6",
                max_tokens=100,
                thinking={"type": "adaptive"},
                messages=request_messages,
            )

        assert len(seen_calls) == 2
        # Attempt 1 went to dep-1 (matched -> thinking kept & unwrapped)
        assert seen_calls[0]["api_key"] == "mock-key-dep-1"
        assert seen_calls[0]["messages"][1]["content"][0]["type"] == "thinking"
        assert seen_calls[0]["messages"][1]["content"][0]["signature"] == "sig_abc_123"

        # Attempt 2 switched to dep-2 after dep-1 cooled down -> thinking stripped proactively!
        assert seen_calls[1]["api_key"] == "mock-key-dep-2"
        dep2_blocks = seen_calls[1]["messages"][1]["content"]
        assert all(
            b["type"] not in ("thinking", "redacted_thinking") for b in dep2_blocks
        )
        # Non-thinking encrypted field unwrapped and preserved
        web_block = next(
            b for b in dep2_blocks if b["type"] == "web_search_tool_result"
        )
        assert web_block["content"][0]["encrypted_content"] == "web_enc_789"
        # Top-level thinking param untouched for adaptive model
        assert seen_calls[1]["thinking"] == {"type": "adaptive"}
        # Final response wrapped with dep-2
        assert _unwrap(response["content"][0]["signature"]) == (
            "dep-2",
            "sig_from_dep_2",
        )

    @pytest.mark.asyncio
    async def test_router_fallback_to_other_model_group_strips_mismatched_thinking(
        self,
    ):
        router = _make_anthropic_router(
            ["dep-primary"],
            model_name="primary-claude",
            num_retries=0,
            fallbacks=[{"primary-claude": ["fallback-claude"]}],
            extra_model_list=[
                {
                    "model_name": "fallback-claude",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-4-6",
                        "api_key": "mock-key-fallback",
                    },
                    "model_info": {"id": "dep-fallback"},
                }
            ],
        )
        seen_calls: List[Dict[str, Any]] = []

        async def fake_upstream(*args, **kwargs):
            api_key = kwargs.get("api_key")
            seen_calls.append(
                {"api_key": api_key, "messages": copy.deepcopy(kwargs["messages"])}
            )
            if api_key == "mock-key-dep-primary":
                raise litellm.InternalServerError(
                    message="primary down",
                    llm_provider="anthropic",
                    model="claude-sonnet-4-6",
                )
            return AnthropicMessagesResponse(**_make_rich_response(sig="sig_fallback"))

        wrapped_hist = wrap_anthropic_messages_response(
            _make_rich_response(), "dep-primary"
        )
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            side_effect=fake_upstream,
        ):
            resp = await router.aanthropic_messages(
                model="primary-claude",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": wrapped_hist["content"]},
                    {"role": "user", "content": "q2"},
                ],
            )

        assert [c["api_key"] for c in seen_calls] == [
            "mock-key-dep-primary",
            "mock-key-fallback",
        ]
        fb_blocks = seen_calls[1]["messages"][1]["content"]
        assert all(
            b["type"] not in ("thinking", "redacted_thinking") for b in fb_blocks
        )
        assert _unwrap(resp["content"][0]["signature"]) == (
            "dep-fallback",
            "sig_fallback",
        )

    @pytest.mark.asyncio
    async def test_router_streaming_roundtrip_and_chat_regression(self):
        router = _make_anthropic_router(["dep-s1", "dep-s2"])

        async def fake_stream_upstream(*args, **kwargs):
            api_key = kwargs.get("api_key")
            return _iter_chunks(
                [
                    _sse_event(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {"id": "msg_s", "content": []},
                        },
                    ),
                    _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            },
                        },
                    ),
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "signature_delta",
                                "signature": f"stream_sig_{api_key}",
                            },
                        },
                    ),
                    _sse_event(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _sse_event("message_stop", {"type": "message_stop"}),
                ]
            )

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            side_effect=fake_stream_upstream,
        ):
            stream = await router.aanthropic_messages(
                model="claude-sonnet-4-6",
                max_tokens=100,
                stream=True,
                messages=[{"role": "user", "content": "stream me"}],
            )
            events = await _collect_sse_events(stream)

        sig_event = next(
            e
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "signature_delta"
        )
        dep_id, orig_sig = _unwrap(sig_event["delta"]["signature"])
        assert dep_id in ("dep-s1", "dep-s2")
        assert orig_sig == f"stream_sig_mock-key-{dep_id}"

        # Regression: chat completions (no litellm_metadata) never gets litellm_metadata or pinned
        check = EncryptedContentAffinityCheck()
        chat_kwargs: Dict[str, Any] = {"metadata": {"tags": ["team-a"]}}
        res = await check.async_filter_deployments(
            model="gpt-4",
            healthy_deployments=[
                {"model_info": {"id": "dep-1"}},
                {"model_info": {"id": "dep-2"}},
            ],
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "signature": _wrap("s", "dep-1")}],
                }
            ],
            request_kwargs=chat_kwargs,
        )
        assert "litellm_metadata" not in chat_kwargs
        assert "_encrypted_content_affinity_pinned" not in chat_kwargs
        assert len(res) == 2


# ---------------------------------------------------------------------------
# 7. Concurrency & Performance tests
# ---------------------------------------------------------------------------


class TestConcurrencyAndPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_multideployment_requests_no_cross_contamination(self):
        dep_ids = ["dep-A", "dep-B", "dep-C"]
        router = _make_anthropic_router(dep_ids)

        async def fake_upstream(*args, **kwargs):
            api_key = kwargs["api_key"]
            dep = api_key.replace("mock-key-", "")
            await asyncio.sleep(0.001)
            return AnthropicMessagesResponse(
                **_make_rich_response(sig=f"sig_{dep}", web_enc=f"web_{dep}")
            )

        async def run_one(idx: int) -> None:
            target_dep = dep_ids[idx % len(dep_ids)]
            # Half the tasks reuse a shared history tagged with target_dep;
            # the other half target an unknown/stale deployment to exercise concurrent stripping.
            origin_dep = target_dep if idx % 2 == 0 else "dep-stale"
            msgs = [
                {"role": "user", "content": f"req-{idx}"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": f"t-{idx}",
                            "signature": _wrap(f"raw_sig_{idx}", origin_dep),
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "encrypted_content": _wrap(
                                        f"raw_web_{idx}", origin_dep
                                    ),
                                }
                            ],
                        },
                        {"type": "text", "text": f"answer-{idx}"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ]
            snapshot = copy.deepcopy(msgs)
            resp = await router.aanthropic_messages(
                model="claude-sonnet-4-6",
                max_tokens=64,
                messages=msgs,
            )
            # Caller messages never mutated under concurrency
            assert msgs == snapshot
            resp_dep, resp_sig = _unwrap(resp["content"][0]["signature"])
            if origin_dep in dep_ids:
                assert resp_dep == target_dep
            else:
                assert resp_dep in dep_ids
            assert resp_sig == f"sig_{resp_dep}"
            web_dep, web_val = _unwrap(
                resp["content"][3]["content"][0]["encrypted_content"]
            )
            assert web_dep == resp_dep
            assert web_val == f"web_{resp_dep}"

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            side_effect=fake_upstream,
        ):
            await asyncio.gather(*(run_one(i) for i in range(60)))

    @pytest.mark.asyncio
    async def test_concurrent_sse_streams_isolated_per_instance(self):
        async def run_stream(idx: int) -> None:
            dep_id = f"deploy-stream-{idx}"
            wire = b"".join(
                [
                    _sse_event(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {"id": f"m_{idx}", "content": []},
                        },
                    ),
                    _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            },
                        },
                    ),
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "signature_delta",
                                "signature": f"s1_{idx}_",
                            },
                        },
                    ),
                    _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "signature_delta",
                                "signature": f"s2_{idx}",
                            },
                        },
                    ),
                    _sse_event(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _sse_event("message_stop", {"type": "message_stop"}),
                ]
            )
            step = 11 + (idx % 13)
            chunks = [wire[i : i + step] for i in range(0, len(wire), step)]
            iterator = AnthropicEncryptedContentAffinityStreamIterator(
                stream=_iter_chunks(chunks, yield_between=True),
                model_id=dep_id,
            )
            events = await _collect_sse_events(iterator)
            sig = "".join(
                e["delta"]["signature"]
                for e in events
                if e.get("type") == "content_block_delta"
                and e.get("delta", {}).get("type") == "signature_delta"
            )
            assert _unwrap(sig) == (dep_id, f"s1_{idx}_s2_{idx}")

        await asyncio.gather(*(run_stream(i) for i in range(60)))

    @pytest.mark.asyncio
    async def test_performance_streaming_fast_path_skips_json_parsing_for_text_deltas(
        self,
    ):
        num_text_deltas = 2000
        chunks = [
            _sse_event(
                "message_start",
                {"type": "message_start", "message": {"id": "msg_perf", "content": []}},
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            *[
                _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": f"tok{i} "},
                    },
                )
                for i in range(num_text_deltas)
            ],
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ]

        json_loads_calls = 0
        real_json_loads = json.loads

        def counting_loads(*args, **kwargs):
            nonlocal json_loads_calls
            json_loads_calls += 1
            return real_json_loads(*args, **kwargs)

        start = time.perf_counter()
        with patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.encrypted_content_affinity.json.loads",
            side_effect=counting_loads,
        ):
            output = [
                c
                async for c in AnthropicEncryptedContentAffinityStreamIterator(
                    _iter_chunks(chunks), "dep-perf"
                )
            ]
        elapsed = time.perf_counter() - start

        assert len(output) == len(chunks)
        # Fast-path must skip json.loads completely on plain text streams
        assert json_loads_calls == 0
        assert elapsed < 1.0, f"Streaming fast-path too slow: {elapsed:.3f}s"

    def test_performance_large_multiturn_history_prepare_and_extract(self):
        # 100 turns (200 messages) with thinking + web search encrypted content
        messages: List[Dict[str, Any]] = []
        for turn in range(100):
            dep = "dep-old" if turn < 90 else "dep-new"
            messages.append({"role": "user", "content": f"question {turn}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "reasoning " * 10,
                            "signature": _wrap(f"sig_{turn}", dep),
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": f"srvtoolu_{turn}",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "encrypted_content": _wrap(f"enc_{turn}", dep),
                                }
                            ],
                        },
                        {"type": "text", "text": f"answer {turn}"},
                    ],
                }
            )

        start = time.perf_counter()
        for _ in range(50):
            latest = extract_latest_model_id_from_anthropic_messages(messages)
            assert latest == "dep-new"
            prepared = prepare_anthropic_messages_for_deployment(messages, "dep-new")
            assert len(prepared) == 200
        elapsed = time.perf_counter() - start

        assert elapsed < 1.5, f"Large history prepare/extract too slow: {elapsed:.3f}s"
