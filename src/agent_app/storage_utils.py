from __future__ import annotations

import re
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_user_id(user_id: str) -> str:
    cleaned = user_id.strip()
    return cleaned or "default"


def looks_sensitive(text: str) -> bool:
    patterns = [
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
        r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b\s*[:=是]\s*\S+",
        r"(?i)\b(access[_ -]?key[_ -]?secret|access[_ -]?token)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def redact_sensitive(text: str) -> str:
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED_SECRET]", text)
    redacted = re.sub(
        r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b(\s*[:=是]\s*)\S+",
        r"\1\2[REDACTED_SECRET]",
        redacted,
    )
    return redacted
