"""Configuration resolution for the Spectron memory provider.

Resolution order for each field: value in ``$HERMES_HOME/spectron.json`` first,
then the environment variable, then ``$HERMES_HOME/.env``, then the built-in
default. Secrets (the API key) are never written to ``spectron.json`` — Hermes
keeps ``secret: True`` fields in ``.env`` — while non-secret settings are
persisted to ``spectron.json`` by :func:`save_config_file`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_FILENAME = "spectron.json"

# Recognised recall strategies.
RECALL_MODES = ("hybrid", "context", "tools")
WRITE_FREQUENCIES = ("turn", "session")


def default_hermes_home(hermes_home: Optional[str] = None) -> str:
    """Best-effort resolution of HERMES_HOME when Hermes hasn't supplied it yet."""
    return (
        hermes_home
        or os.environ.get("HERMES_HOME")
        or str(Path.home() / ".hermes")
    )


@dataclass
class SpectronConfig:
    """Resolved settings for a Spectron client + provider behaviour."""

    endpoint: Optional[str] = None
    context: Optional[str] = None
    api_key: Optional[str] = None
    recall_mode: str = "hybrid"
    write_frequency: str = "turn"
    top_k: int = 5
    default_scope: Optional[str] = None
    consolidate_on_end: bool = True
    timeout: float = 30.0
    max_retries: int = 3

    def is_configured(self) -> bool:
        """True when the minimum needed to talk to Spectron is present."""
        return bool(self.endpoint and self.context and self.api_key)


def _config_path(hermes_home: Optional[str]) -> Path:
    return Path(default_hermes_home(hermes_home)) / CONFIG_FILENAME


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _read_env_file(hermes_home: Optional[str]) -> Dict[str, str]:
    """Parse ``$HERMES_HOME/.env`` into a dict. Missing/unreadable file → ``{}``.

    Hermes writes secrets here, but whether they are exported into ``os.environ``
    depends on the host process. Reading the file directly means a key saved by
    ``hermes memory setup`` resolves even when nothing loaded it into the
    environment — otherwise a correctly configured provider reports itself
    unavailable.
    """
    path = Path(default_hermes_home(hermes_home)) / ".env"
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _pick(
    file_cfg: Dict[str, Any],
    key: str,
    env_var: str,
    default: Any,
    env_file: Optional[Dict[str, str]] = None,
) -> Any:
    if key in file_cfg and file_cfg[key] not in (None, ""):
        return file_cfg[key]
    env_val = os.environ.get(env_var)
    if env_val not in (None, ""):
        return env_val
    # A deliberate export outranks the file, so this comes last.
    if env_file:
        file_val = env_file.get(env_var)
        if file_val not in (None, ""):
            return file_val
    return default


def load_config(hermes_home: Optional[str] = None) -> SpectronConfig:
    """Load and resolve configuration from file, environment, then defaults."""
    file_cfg = _read_json(_config_path(hermes_home))
    env_file = _read_env_file(hermes_home)

    def _int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _bool(value: Any, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return fallback

    recall_mode = str(_pick(file_cfg, "recall_mode", "SPECTRON_RECALL_MODE", "hybrid"))
    if recall_mode not in RECALL_MODES:
        recall_mode = "hybrid"

    write_frequency = str(
        _pick(file_cfg, "write_frequency", "SPECTRON_WRITE_FREQUENCY", "turn")
    )
    if write_frequency not in WRITE_FREQUENCIES:
        write_frequency = "turn"

    return SpectronConfig(
        # endpoint/context also land in .env when installed via
        # `hermes plugins install` (which prompts for requires_env), so they get
        # the same fallback as the API key.
        endpoint=_pick(file_cfg, "endpoint", "SPECTRON_ENDPOINT", None, env_file),
        context=_pick(file_cfg, "context", "SPECTRON_CONTEXT", None, env_file),
        api_key=_pick(file_cfg, "api_key", "SPECTRON_API_KEY", None, env_file),
        recall_mode=recall_mode,
        write_frequency=write_frequency,
        top_k=_int(_pick(file_cfg, "top_k", "SPECTRON_TOP_K", 5), 5),
        default_scope=_pick(file_cfg, "default_scope", "SPECTRON_DEFAULT_SCOPE", None),
        consolidate_on_end=_bool(
            _pick(file_cfg, "consolidate_on_end", "SPECTRON_CONSOLIDATE_ON_END", True),
            True,
        ),
        timeout=_float(_pick(file_cfg, "timeout", "SPECTRON_TIMEOUT", 30.0), 30.0),
        max_retries=_int(_pick(file_cfg, "max_retries", "SPECTRON_MAX_RETRIES", 3), 3),
    )


# Non-secret keys persisted to spectron.json. The API key is intentionally
# excluded — it belongs in the environment / .env.
_PERSISTED_KEYS = (
    "endpoint",
    "context",
    "recall_mode",
    "write_frequency",
    "top_k",
    "default_scope",
    "consolidate_on_end",
    "timeout",
    "max_retries",
)


def save_config_file(values: Dict[str, Any], hermes_home: Optional[str]) -> Path:
    """Persist non-secret settings to ``$HERMES_HOME/spectron.json`` atomically."""
    path = _config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json(path)
    for key in _PERSISTED_KEYS:
        if key in values and values[key] not in (None, ""):
            existing[key] = values[key]

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".spectron-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def config_schema() -> List[Dict[str, Any]]:
    """Field descriptors driving ``hermes memory setup`` (see MemoryProvider ABC)."""
    return [
        {
            "key": "api_key",
            "description": "Spectron API key",
            "secret": True,
            "required": True,
            "env_var": "SPECTRON_API_KEY",
            "url": "https://surrealdb.com/platform/spectron",
        },
        {
            "key": "endpoint",
            "description": "Spectron endpoint origin, e.g. https://your-instance.spectron.dev",
            "required": True,
            "env_var": "SPECTRON_ENDPOINT",
        },
        {
            "key": "context",
            "description": "Spectron context this agent's memory is pinned to",
            "required": True,
            "env_var": "SPECTRON_CONTEXT",
        },
        {
            "key": "recall_mode",
            "description": "How memory is surfaced each turn",
            "default": "hybrid",
            "choices": list(RECALL_MODES),
            "env_var": "SPECTRON_RECALL_MODE",
        },
        {
            "key": "write_frequency",
            "description": "When completed turns are written back to Spectron",
            "default": "turn",
            "choices": list(WRITE_FREQUENCIES),
            "env_var": "SPECTRON_WRITE_FREQUENCY",
        },
        {
            "key": "top_k",
            "description": "Number of memories to recall per turn",
            "default": 5,
            "env_var": "SPECTRON_TOP_K",
        },
        {
            "key": "default_scope",
            "description": "Optional default scope for writes / lens for reads, e.g. user/tobie",
            "env_var": "SPECTRON_DEFAULT_SCOPE",
        },
    ]
