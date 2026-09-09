"""Fallback ``MemoryProvider`` base for running outside a Hermes install.

When this package runs inside Hermes, :class:`agent.memory_provider.MemoryProvider`
is importable and is used as the real base class. Outside Hermes (e.g. unit tests,
type checking, or `pip install` verification), that module is absent, so we provide
a minimal stand-in with the same method surface and default no-op behaviour.

The real ABC is authoritative — keep this mirror in sync with
``agent/memory_provider.py`` in NousResearch/hermes-agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryProvider(ABC):  # pragma: no cover - exercised only outside Hermes
    """Minimal stand-in mirroring Hermes' MemoryProvider abstract base class."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]: ...

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        pass

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    def shutdown(self) -> None:
        pass

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass

    def backup_paths(self) -> List[str]:
        return []
