from __future__ import annotations

from dataclasses import asdict, dataclass

from .storage_utils import looks_sensitive


@dataclass(frozen=True)
class TurnRoutingDecision:
    should_retrieve: bool
    retrieve_reason: str
    should_learn: bool
    learn_reason: str

    def to_state(self) -> dict[str, bool | str]:
        return asdict(self)


class TurnRoutingPolicy:
    def evaluate(self, user_text: str) -> TurnRoutingDecision:
        retrieve, retrieve_reason = self._retrieve_decision(user_text)
        learn, learn_reason = self._learn_decision(user_text)
        return TurnRoutingDecision(
            should_retrieve=retrieve,
            retrieve_reason=retrieve_reason,
            should_learn=learn,
            learn_reason=learn_reason,
        )

    def should_learn(self, user_text: str) -> bool:
        return self.evaluate(user_text).should_learn

    def should_retrieve(self, user_text: str) -> bool:
        return self.evaluate(user_text).should_retrieve

    def _learn_decision(self, user_text: str) -> tuple[bool, str]:
        text = user_text.strip()
        if not text:
            return False, "empty"
        if looks_sensitive(text):
            return False, "sensitive"
        if text.startswith("/"):
            return False, "command"
        if text.lower() in _LOW_SIGNAL_PHRASES:
            return False, "low_signal"
        if len(text) <= 2:
            return False, "too_short"
        if any(phrase in text for phrase in _RECALL_PHRASES):
            return False, "recall_turn"
        return True, "default_learn"

    def _retrieve_decision(self, user_text: str) -> tuple[bool, str]:
        text = user_text.strip()
        if not text:
            return False, "empty"
        if looks_sensitive(text):
            return False, "sensitive"
        if text.startswith("/"):
            return False, "command"
        if text.lower() in _LOW_SIGNAL_PHRASES:
            return False, "low_signal"
        if len(text) <= 2:
            return False, "too_short"
        return True, "default_retrieve"


_LOW_SIGNAL_PHRASES = {
    "你好",
    "您好",
    "嗨",
    "hi",
    "hello",
    "thanks",
    "thank you",
    "谢谢",
    "好的",
    "ok",
    "okay",
}

_RECALL_PHRASES = (
    "你还记得",
    "你记得吗",
    "还记得吗",
    "还记得我的",
    "记得我的",
)
