"""
Encrypted-content-aware deployment affinity utilities for Anthropic `/v1/messages`.

Embeds the originating deployment's ``model_info.id`` into encrypted and
signature fields produced in assistant `/v1/messages` responses (both
non-streaming JSON and streaming SSE), using the same stateless prefix format
as `/v1/responses`:

    ``litellm_enc:{base64("model_id:{model_id}")};{original_value}``

On follow-up requests:
1. Routing (`EncryptedContentAffinityCheck`) extracts the latest deployment ID
   from ``role == "assistant"`` messages to prefer the originating deployment
   when healthy (ignoring ``user`` / ``system`` / ``tool_result`` payloads).
2. Before forwarding upstream (including on initial miss, Router retries, and
   Router fallbacks), `prepare_anthropic_messages_for_deployment` produces a
   copy-on-write request message list:
   - Matching deployment: unwraps prefixes and keeps thinking/encrypted fields.
   - Mismatched deployment: strips only mismatched `thinking` /
     `redacted_thinking` blocks (preserving earlier blocks signed by the
     selected deployment and its prompt cache; omitting assistant messages
     whose `content` list becomes empty), strips foreign non-thinking
     `signature` / `thought_signature` fields (e.g. Gemini
     `tool_use.provider_specific_fields.signature` / `thought_signature`),
     leaves the top-level `thinking` param untouched, and unwraps remaining
     `encrypted_*` fields while never modifying `tool_use.input`,
     `server_tool_use.input`, or media `source` payloads.
"""

import codecs
import inspect
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from litellm.responses.utils import ResponsesAPIRequestUtils

_WRAPPED_PREFIX = "litellm_enc:"
_ENCRYPTED_KEY_PREFIX = "encrypted_"
_SIGNATURE_FIELD_NAMES = frozenset({"signature", "thought_signature"})
_SKIP_SSE_DELTA_TYPES = frozenset({"text_delta", "thinking_delta", "input_json_delta"})
_SSE_ENCRYPTED_TRIGGER_TOKENS = (
    "signature",
    "redacted_thinking",
    "encrypted_",
    "thought_signature",
)


def _get_thinking_block_field(block: Dict[str, Any]) -> Optional[str]:
    """Return the encrypted/signature field name for a thinking block, or None."""
    block_type = block.get("type")
    if block_type == "thinking":
        return "signature"
    if block_type == "redacted_thinking":
        return "data"
    return None


def _is_encrypted_field_key(key: str, block_type: Any) -> bool:
    """Return True if ``key`` on a dict with ``block_type`` is an encrypted/signature field.

    ``data`` is ONLY treated as encrypted on ``redacted_thinking`` blocks so
    media payloads like ``image.source.data`` / ``document.source.data`` are
    never modified.
    """
    if key == "data":
        return block_type == "redacted_thinking"
    return key.startswith(_ENCRYPTED_KEY_PREFIX) or key in _SIGNATURE_FIELD_NAMES


def _should_skip_child_key(key: str, block_type: Any) -> bool:
    """Return True for business/media payload subtrees that must never be rewritten.

    Prevents corrupting tool arguments (e.g. ``tool_use.input.signature`` or
    ``tool_use.input.encrypted_token``) and skips media ``source`` payloads.
    """
    if key == "source":
        return True
    if (
        key == "input"
        and isinstance(block_type, str)
        and block_type.endswith("tool_use")
    ):
        return True
    return False


def _wrap_value(value: str, model_id: str) -> str:
    return ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
        value, model_id
    )


def _unwrap_value(value: str) -> Tuple[Optional[str], str]:
    return ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(value)


def _extract_latest_model_id_from_node(node: Any) -> Optional[str]:
    """Recursively scan ``node`` in reverse order for the latest wrapped model_id."""
    if isinstance(node, list):
        for item in reversed(node):
            model_id = _extract_latest_model_id_from_node(item)
            if model_id is not None:
                return model_id
        return None

    if isinstance(node, dict):
        block_type = node.get("type")
        for key, value in reversed(list(node.items())):
            if isinstance(key, str) and _should_skip_child_key(key, block_type):
                continue
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and value.startswith(_WRAPPED_PREFIX)
                and _is_encrypted_field_key(key, block_type)
            ):
                model_id, _ = _unwrap_value(value)
                if model_id is not None:
                    return model_id
            elif isinstance(value, (dict, list)):
                model_id = _extract_latest_model_id_from_node(value)
                if model_id is not None:
                    return model_id

    return None


def extract_latest_model_id_from_anthropic_messages(messages: Any) -> Optional[str]:
    """Extract the most recently embedded deployment ``model_id`` from assistant `/v1/messages`.

    Only inspects ``role == "assistant"`` content blocks so ``user`` / ``system``
    / ``tool_result`` payloads cannot hijack deployment routing.
    """
    if not isinstance(messages, list) or not messages:
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        model_id = _extract_latest_model_id_from_node(content)
        if model_id is not None:
            return model_id
    return None


def _contains_wrapped_marker(node: Any) -> bool:
    """Fast read-only scan checking if any encrypted/signature field carries ``litellm_enc:``."""
    if isinstance(node, list):
        return any(_contains_wrapped_marker(item) for item in node)

    if isinstance(node, dict):
        block_type = node.get("type")
        for key, value in node.items():
            if isinstance(key, str) and _should_skip_child_key(key, block_type):
                continue
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and value.startswith(_WRAPPED_PREFIX)
                and _is_encrypted_field_key(key, block_type)
            ):
                return True
            if isinstance(value, (dict, list)) and _contains_wrapped_marker(value):
                return True

    return False


def _assistant_messages_contain_wrapped_marker(messages: List[Dict[str, Any]]) -> bool:
    """Return True if any assistant message content contains a ``litellm_enc:`` field."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list) and _contains_wrapped_marker(content):
            return True
    return False


def _prepare_non_thinking_node(
    node: Any,
    normalized_current_id: Optional[str],
) -> Tuple[Any, bool]:
    """Copy-on-write prepare of non-thinking blocks for the target deployment.

    - Unwraps all valid ``litellm_enc:`` encrypted/signature fields.
    - When ``normalized_current_id`` is set and a foreign ``signature`` /
      ``thought_signature`` field (e.g. ``tool_use.provider_specific_fields``)
      belongs to a different deployment, removes that signature key (and drops
      ``provider_specific_fields`` if it becomes empty) to avoid cross-provider
      400 errors such as Gemini ``Corrupted thought signature``.
    - Preserves ``encrypted_*`` fields (unwrapped) and never touches ``input``
      or ``source`` payloads.
    """
    if isinstance(node, list):
        changed = False
        new_list: List[Any] = []
        for item in node:
            prepared_item, item_changed = _prepare_non_thinking_node(
                item, normalized_current_id
            )
            if item_changed:
                changed = True
            new_list.append(prepared_item)
        return (new_list, True) if changed else (node, False)

    if isinstance(node, dict):
        block_type = node.get("type")
        updates: Dict[str, Any] = {}
        dropped_keys: Set[str] = set()
        for key, value in node.items():
            if isinstance(key, str) and _should_skip_child_key(key, block_type):
                continue
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and value.startswith(_WRAPPED_PREFIX)
                and _is_encrypted_field_key(key, block_type)
            ):
                origin_model_id, unwrapped_val = _unwrap_value(value)
                if origin_model_id is not None:
                    if (
                        key in _SIGNATURE_FIELD_NAMES
                        and normalized_current_id is not None
                        and origin_model_id != normalized_current_id
                    ):
                        dropped_keys.add(key)
                    else:
                        updates[key] = unwrapped_val
            elif isinstance(value, (dict, list)):
                prepared_child, child_changed = _prepare_non_thinking_node(
                    value, normalized_current_id
                )
                if child_changed:
                    if (
                        key == "provider_specific_fields"
                        and isinstance(prepared_child, dict)
                        and len(prepared_child) == 0
                    ):
                        dropped_keys.add(key)
                    else:
                        updates[key] = prepared_child

        if not updates and not dropped_keys:
            return node, False

        return {
            k: updates[k] if k in updates else v
            for k, v in node.items()
            if k not in dropped_keys
        }, True

    return node, False


def _wrap_node(node: Any, model_id: str) -> Tuple[Any, bool]:
    """Copy-on-write wrap of all non-empty encrypted/signature fields within ``node``."""
    if isinstance(node, list):
        changed = False
        new_list: List[Any] = []
        for item in node:
            wrapped_item, item_changed = _wrap_node(item, model_id)
            if item_changed:
                changed = True
            new_list.append(wrapped_item)
        return (new_list, True) if changed else (node, False)

    if isinstance(node, dict):
        block_type = node.get("type")
        updates: Dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(key, str) and _should_skip_child_key(key, block_type):
                continue
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and value
                and _is_encrypted_field_key(key, block_type)
            ):
                updates[key] = _wrap_value(value, model_id)
            elif isinstance(value, (dict, list)):
                wrapped_child, child_changed = _wrap_node(value, model_id)
                if child_changed:
                    updates[key] = wrapped_child
        if updates:
            return {**node, **updates}, True
        return node, False

    return node, False


def prepare_anthropic_messages_for_deployment(
    messages: List[Dict[str, Any]],
    current_model_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Prepare `/v1/messages` request ``messages`` for the selected deployment.

    Never mutates ``messages`` in-place so Router retries and fallbacks always
    inspect the original deployment markers. Only inspects ``role == "assistant"``
    content blocks; ``user`` and ``system`` messages pass through untouched.
    """
    if not isinstance(messages, list) or not messages:
        return messages
    if not _assistant_messages_contain_wrapped_marker(messages):
        return messages

    normalized_current_id = (
        str(current_model_id)
        if current_model_id is not None and str(current_model_id) != ""
        else None
    )

    prepared_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            prepared_messages.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, list):
            prepared_messages.append(msg)
            continue

        new_content: List[Any] = []
        content_changed = False
        msg_from_other_deployment: Optional[bool] = None
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            thinking_field = _get_thinking_block_field(block)
            if thinking_field is not None:
                raw_val = block.get(thinking_field)
                if isinstance(raw_val, str) and raw_val.startswith(_WRAPPED_PREFIX):
                    origin_model_id, unwrapped_val = _unwrap_value(raw_val)
                    if origin_model_id is not None:
                        if (
                            normalized_current_id is not None
                            and origin_model_id != normalized_current_id
                        ):
                            content_changed = True
                            continue
                        content_changed = True
                        new_content.append({**block, thinking_field: unwrapped_val})
                        continue
                elif normalized_current_id is not None and (
                    not isinstance(raw_val, str) or raw_val == ""
                ):
                    # Unsigned/empty thinking block inside an assistant turn from
                    # another deployment must also be stripped so upstream doesn't
                    # reject an empty-signature thinking block on failover.
                    if msg_from_other_deployment is None:
                        msg_origin = _extract_latest_model_id_from_node(content)
                        msg_from_other_deployment = (
                            msg_origin is not None
                            and msg_origin != normalized_current_id
                        )
                    if msg_from_other_deployment:
                        content_changed = True
                        continue
                new_content.append(block)
                continue

            prepared_block, block_changed = _prepare_non_thinking_node(
                block, normalized_current_id
            )
            if block_changed:
                content_changed = True
            new_content.append(prepared_block)

        if content_changed and len(new_content) == 0:
            # Omit assistant messages whose content array became empty after stripping thinking blocks
            continue

        prepared_messages.append(
            {**msg, "content": new_content} if content_changed else msg
        )

    return prepared_messages


def wrap_anthropic_messages_response(response: Any, model_id: Optional[str]) -> Any:
    """Wrap encrypted/signature fields in a non-streaming `/v1/messages` response."""
    if not model_id or not isinstance(response, dict):
        return response
    content = response.get("content")
    if not isinstance(content, list):
        return response
    wrapped_content, changed = _wrap_node(content, str(model_id))
    if not changed:
        return response
    return {**response, "content": wrapped_content}


def _wrap_sse_event_dict(
    event_dict: Dict[str, Any],
    model_id: str,
    signature_wrapped_indices: Set[int],
) -> Tuple[Dict[str, Any], bool]:
    """Wrap encrypted/signature fields in a parsed Anthropic SSE JSON event dict."""
    event_type = event_dict.get("type")

    if event_type == "message_start":
        signature_wrapped_indices.clear()
        message_obj = event_dict.get("message")
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            if isinstance(content, list) and content:
                wrapped_content, changed = _wrap_node(content, model_id)
                if changed:
                    return {
                        **event_dict,
                        "message": {**message_obj, "content": wrapped_content},
                    }, True
        return event_dict, False

    if event_type == "content_block_start":
        index = event_dict.get("index")
        if isinstance(index, int):
            signature_wrapped_indices.discard(index)
        content_block = event_dict.get("content_block")
        if isinstance(content_block, dict):
            wrapped_block, changed = _wrap_node(content_block, model_id)
            if content_block.get("type") == "thinking":
                sig = content_block.get("signature")
                if isinstance(sig, str) and sig and isinstance(index, int):
                    signature_wrapped_indices.add(index)
            if changed:
                return {**event_dict, "content_block": wrapped_block}, True
        return event_dict, False

    if event_type == "content_block_delta":
        index = event_dict.get("index")
        delta = event_dict.get("delta")
        if isinstance(delta, dict):
            delta_type = delta.get("type")
            if delta_type in _SKIP_SSE_DELTA_TYPES:
                return event_dict, False
            if delta_type == "signature_delta":
                sig = delta.get("signature")
                if isinstance(sig, str) and sig:
                    if not isinstance(index, int) or (
                        index not in signature_wrapped_indices
                    ):
                        if isinstance(index, int):
                            signature_wrapped_indices.add(index)
                        return {
                            **event_dict,
                            "delta": {
                                **delta,
                                "signature": _wrap_value(sig, model_id),
                            },
                        }, True
                return event_dict, False

            wrapped_delta, changed = _wrap_node(delta, model_id)
            if changed:
                return {**event_dict, "delta": wrapped_delta}, True
        return event_dict, False

    if event_type == "content_block_stop":
        index = event_dict.get("index")
        if isinstance(index, int):
            signature_wrapped_indices.discard(index)
        return event_dict, False

    return event_dict, False


def _split_next_sse_frame(buffer: str) -> Optional[Tuple[str, str, str]]:
    """Split the earliest complete SSE frame from ``buffer``.

    Returns ``(frame_body, separator, remaining)`` or ``None`` if incomplete.
    """
    lf_idx = buffer.find("\n\n")
    crlf_idx = buffer.find("\r\n\r\n")

    if lf_idx == -1 and crlf_idx == -1:
        return None
    if lf_idx != -1 and (crlf_idx == -1 or lf_idx < crlf_idx):
        return buffer[:lf_idx], "\n\n", buffer[lf_idx + 2 :]
    return buffer[:crlf_idx], "\r\n\r\n", buffer[crlf_idx + 4 :]


def _rewrite_sse_frame(
    frame_body: str,
    model_id: str,
    signature_wrapped_indices: Set[int],
) -> str:
    """Inspect and rewrite a single SSE frame body if it carries encrypted fields."""
    if not frame_body:
        return frame_body

    has_encrypted_token = any(
        token in frame_body for token in _SSE_ENCRYPTED_TRIGGER_TOKENS
    )
    if not has_encrypted_token:
        if "message_start" in frame_body:
            signature_wrapped_indices.clear()
            return frame_body
        if not signature_wrapped_indices or (
            "content_block_start" not in frame_body
            and "content_block_stop" not in frame_body
        ):
            return frame_body

    line_ending = "\r\n" if "\r\n" in frame_body else "\n"
    lines = frame_body.splitlines()

    data_lines: List[str] = []
    other_lines: List[str] = []
    data_insert_pos: Optional[int] = None

    for line in lines:
        if line.startswith("data:"):
            if data_insert_pos is None:
                data_insert_pos = len(other_lines)
            payload = line[5:]
            if payload.startswith(" "):
                payload = payload[1:]
            data_lines.append(payload)
        else:
            other_lines.append(line)

    if not data_lines:
        return frame_body

    data_str = "\n".join(data_lines)
    try:
        parsed = json.loads(data_str)
    except (json.JSONDecodeError, ValueError):
        return frame_body

    if not isinstance(parsed, dict):
        return frame_body

    wrapped_event, changed = _wrap_sse_event_dict(
        event_dict=parsed,
        model_id=model_id,
        signature_wrapped_indices=signature_wrapped_indices,
    )
    if not changed:
        return frame_body

    rewritten_data_line = f"data: {json.dumps(wrapped_event, ensure_ascii=False)}"
    insert_at = data_insert_pos if data_insert_pos is not None else len(other_lines)
    rebuilt_lines = (
        other_lines[:insert_at] + [rewritten_data_line] + other_lines[insert_at:]
    )
    return line_ending.join(rebuilt_lines)


class AnthropicEncryptedContentAffinityStreamIterator:
    """Chunk-boundary-safe async SSE wrapper embedding ``model_id`` into encrypted fields."""

    def __init__(self, stream: AsyncIterator[Any], model_id: str) -> None:
        self._inner = stream.__aiter__()
        self._model_id = str(model_id)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer: str = ""
        self._signature_wrapped_indices: Set[int] = set()
        self._emit_bytes: bool = True
        self._exhausted: bool = False

    def __aiter__(self) -> "AnthropicEncryptedContentAffinityStreamIterator":
        return self

    async def aclose(self) -> None:
        """Close the wrapped async stream to promptly release upstream resources."""
        self._exhausted = True
        self._buffer = ""
        close_fn = getattr(self._inner, "aclose", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result

    def _drain_complete_frames(self) -> str:
        output_parts: List[str] = []
        while True:
            split_result = _split_next_sse_frame(self._buffer)
            if split_result is None:
                break
            frame_body, separator, remaining = split_result
            self._buffer = remaining
            rewritten = _rewrite_sse_frame(
                frame_body=frame_body,
                model_id=self._model_id,
                signature_wrapped_indices=self._signature_wrapped_indices,
            )
            output_parts.append(rewritten)
            output_parts.append(separator)
        return "".join(output_parts)

    def _encode_output(self, text: str) -> Any:
        return text.encode("utf-8") if self._emit_bytes else text

    async def __anext__(self) -> Any:
        if self._exhausted:
            raise StopAsyncIteration

        while True:
            try:
                chunk = await self._inner.__anext__()
            except StopAsyncIteration:
                self._exhausted = True
                tail = self._decoder.decode(b"", final=True)
                if tail:
                    self._buffer += tail
                if self._buffer:
                    # Flush complete frames first, then any trailing partial frame without \n\n
                    flushed = self._drain_complete_frames()
                    if self._buffer:
                        flushed += _rewrite_sse_frame(
                            frame_body=self._buffer,
                            model_id=self._model_id,
                            signature_wrapped_indices=self._signature_wrapped_indices,
                        )
                        self._buffer = ""
                    if flushed:
                        return self._encode_output(flushed)
                raise StopAsyncIteration

            if isinstance(chunk, (bytes, bytearray)):
                self._emit_bytes = True
                self._buffer += self._decoder.decode(bytes(chunk), final=False)
                emitted = self._drain_complete_frames()
                if emitted:
                    return emitted.encode("utf-8")
                continue

            if isinstance(chunk, str):
                self._emit_bytes = False
                self._buffer += chunk
                emitted = self._drain_complete_frames()
                if emitted:
                    return emitted
                continue

            if isinstance(chunk, dict):
                wrapped_dict, _ = _wrap_sse_event_dict(
                    event_dict=chunk,
                    model_id=self._model_id,
                    signature_wrapped_indices=self._signature_wrapped_indices,
                )
                return wrapped_dict

            return chunk


def get_anthropic_affinity_context(
    kwargs: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Extract ``(affinity_enabled, current_model_id)`` from `/v1/messages` kwargs."""
    litellm_metadata = kwargs.get("litellm_metadata")
    metadata_dict = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    affinity_enabled = bool(metadata_dict.get("encrypted_content_affinity_enabled"))

    model_info = metadata_dict.get("model_info")
    raw_model_id = model_info.get("id") if isinstance(model_info, dict) else None
    if raw_model_id is None or str(raw_model_id) == "":
        fallback_info = kwargs.get("model_info")
        if isinstance(fallback_info, dict):
            raw_model_id = fallback_info.get("id")
    current_model_id = (
        str(raw_model_id)
        if raw_model_id is not None and str(raw_model_id) != ""
        else None
    )
    return affinity_enabled, current_model_id


def apply_anthropic_response_affinity_wrapping(
    response: Any,
    kwargs: Dict[str, Any],
) -> Any:
    """Wrap `/v1/messages` response or async stream when affinity is active."""
    if kwargs.get("_agentic_loop_depth", 0):
        return response

    affinity_enabled, current_model_id = get_anthropic_affinity_context(kwargs)
    if not affinity_enabled or not current_model_id:
        return response

    if hasattr(response, "__aiter__"):
        return AnthropicEncryptedContentAffinityStreamIterator(
            stream=response,
            model_id=current_model_id,
        )

    if isinstance(response, dict):
        return wrap_anthropic_messages_response(
            response=response,
            model_id=current_model_id,
        )

    return response
