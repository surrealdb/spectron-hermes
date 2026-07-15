"""Tool schemas exposed to the agent and dispatch to the Spectron SDK.

Schemas use the OpenAI function-calling shape (``name`` / ``description`` /
``parameters``), matching Hermes' bundled memory providers. ``dispatch`` maps a
tool call to a Spectron client method and returns a JSON-serialisable dict.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from .config import SpectronConfig

RECALL_SCHEMA: Dict[str, Any] = {
    "name": "spectron_recall",
    "description": (
        "Search long-term memory in Spectron for facts relevant to a query, "
        "ranked across semantic, lexical, graph and temporal signals. Use this "
        "to retrieve what is known about a person, project, or topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search memory for."},
            "k": {
                "type": "integer",
                "description": "Max number of memories to return (defaults to the configured top_k).",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA: Dict[str, Any] = {
    "name": "spectron_remember",
    "description": (
        "Store a durable fact in Spectron memory. Prefer concise, self-contained "
        "statements. Spectron versions facts tri-temporally and never overwrites "
        "history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact to remember."},
            "scope": {
                "type": "string",
                "description": "Optional scope for the fact, e.g. 'user/tobie'. Defaults to the configured scope.",
            },
        },
        "required": ["text"],
    },
}

CONTEXT_SCHEMA: Dict[str, Any] = {
    "name": "spectron_context",
    "description": (
        "Ask Spectron to synthesise an answer from memory for a question, rather "
        "than returning raw hits. Use when you want a summarised, reasoned view."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to answer from memory."},
            "k": {
                "type": "integer",
                "description": "Max memories to consider (defaults to the configured top_k).",
            },
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA: Dict[str, Any] = {
    "name": "spectron_forget",
    "description": (
        "Forget memories matching a query. By default this supersedes them "
        "(kept as history, marked no longer valid). Set purge=true to hard-delete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to forget."},
            "purge": {
                "type": "boolean",
                "description": "Hard-delete instead of superseding. Defaults to false.",
            },
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA: Dict[str, Any] = {
    "name": "spectron_reflect",
    "description": (
        "Run a reflection over memory to derive higher-level insights about a "
        "topic. Set persist=true to write the reflection back into memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to reflect on."},
            "persist": {
                "type": "boolean",
                "description": "Persist the reflection into memory. Defaults to false.",
            },
        },
        "required": ["query"],
    },
}

UPLOAD_SCHEMA: Dict[str, Any] = {
    "name": "spectron_upload",
    "description": (
        "Ingest a document from a local file path into Spectron's knowledge "
        "memory so its contents become recallable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Local filesystem path to the document."},
            "title": {"type": "string", "description": "Optional human-readable title."},
        },
        "required": ["path"],
    },
}

ALL_SCHEMAS: List[Dict[str, Any]] = [
    RECALL_SCHEMA,
    REMEMBER_SCHEMA,
    CONTEXT_SCHEMA,
    FORGET_SCHEMA,
    REFLECT_SCHEMA,
    UPLOAD_SCHEMA,
]

TOOL_NAMES = frozenset(schema["name"] for schema in ALL_SCHEMAS)


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of Spectron SDK response objects to plain JSON data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # Pydantic v2 / v1 models.
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return to_jsonable(method())
            except Exception:
                pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "__dict__"):
        return {
            k: to_jsonable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def dispatch(
    client: Any,
    config: SpectronConfig,
    default_scope: Optional[str],
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    """Execute a single tool call against the Spectron client.

    Returns a JSON-serialisable result. Raises on unknown tool names and lets
    SDK exceptions propagate to the caller (the provider handles/logs them).
    """
    args = args or {}

    if tool_name == "spectron_recall":
        k = args.get("k") or config.top_k
        return to_jsonable(client.recall(args["query"], k=k))

    if tool_name == "spectron_remember":
        scope = args.get("scope") or default_scope
        if scope:
            return to_jsonable(client.remember(args["text"], scopes=scope))
        return to_jsonable(client.remember(args["text"]))

    if tool_name == "spectron_context":
        k = args.get("k") or config.top_k
        return to_jsonable(client.query_context(args["query"], k=k))

    if tool_name == "spectron_forget":
        return to_jsonable(client.forget(args["query"], purge=bool(args.get("purge", False))))

    if tool_name == "spectron_reflect":
        return to_jsonable(client.reflect(args["query"], persist=bool(args.get("persist", False))))

    if tool_name == "spectron_upload":
        title = args.get("title")
        if title:
            return to_jsonable(client.documents.upload(args["path"], title=title))
        return to_jsonable(client.documents.upload(args["path"]))

    raise KeyError(tool_name)
