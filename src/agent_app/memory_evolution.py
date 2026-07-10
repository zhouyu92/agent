from __future__ import annotations

import os

CORRECTION_MARKERS: tuple[str, ...] = (
    "不是",
    "而是",
    "改成",
    "更准确地说",
    "实际上",
    "以后以",
    "以后回答",
)

WEAK_SIGNAL_PHRASES: tuple[str, ...] = (
    "哈哈",
    "哈哈哈",
    "哈哈哈哈",
    "呵呵",
    "嗯嗯",
    "好的",
    "收到",
)

PREFERENCE_SHIFT_MARKERS: tuple[str, ...] = (
    "现在更喜欢",
    "现在喜欢",
    "更喜欢",
    "改喜欢",
)

CONTEXTUAL_PREFERENCE_MARKERS: tuple[str, ...] = (
    "写代码时",
    "工作时",
    "闲聊时",
    "讨论方案时",
    "在代码里",
)


def looks_like_correction(text: str) -> bool:
    lowered = text.strip()
    return any(marker in lowered for marker in CORRECTION_MARKERS)


def looks_like_weak_signal(text: str) -> bool:
    normalized = text.strip()
    if normalized in WEAK_SIGNAL_PHRASES:
        return True
    if len(normalized) <= 6 and len(set(normalized)) <= 2:
        return True
    return False


def looks_like_preference_shift(candidate_content: str, matched_content: str) -> bool:
    candidate = candidate_content.strip()
    matched = matched_content.strip()
    if candidate == matched:
        return False
    if "喜欢" not in candidate or "喜欢" not in matched:
        return False
    if has_contextual_preference_scope(candidate) and not has_contextual_preference_scope(matched):
        return False
    return any(marker in candidate for marker in PREFERENCE_SHIFT_MARKERS)


def has_contextual_preference_scope(text: str) -> bool:
    normalized = text.strip()
    return any(marker in normalized for marker in CONTEXTUAL_PREFERENCE_MARKERS)


def looks_like_fact_correction(candidate_content: str, matched_content: str) -> bool:
    candidate = candidate_content.strip()
    matched = matched_content.strip()
    if candidate == matched:
        return False
    shared_prefix = os.path.commonprefix([candidate, matched])
    if len(shared_prefix) < 4:
        return False
    shared_suffix = os.path.commonprefix([candidate[::-1], matched[::-1]])
    return len(shared_suffix) >= 1


def choose_evolution_action(candidate_content: str, matched_content: str) -> tuple[str, str]:
    if looks_like_correction(candidate_content):
        return "revise", "correction_phrase"
    if looks_like_preference_shift(candidate_content, matched_content):
        return "revise", "preference_shift"
    if looks_like_fact_correction(candidate_content, matched_content):
        return "revise", "fact_correction"
    if candidate_content.strip() == matched_content.strip():
        return "ignore", "no_new_information"
    return "reinforce", "confirmed_existing_memory"
