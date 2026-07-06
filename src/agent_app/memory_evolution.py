from __future__ import annotations

CORRECTION_MARKERS: tuple[str, ...] = (
    "不是",
    "而是",
    "改成",
    "更准确地说",
    "实际上",
    "以后以",
    "以后回答",
)


def looks_like_correction(text: str) -> bool:
    lowered = text.strip()
    return any(marker in lowered for marker in CORRECTION_MARKERS)


def choose_evolution_action(candidate_content: str, matched_content: str) -> tuple[str, str]:
    if looks_like_correction(candidate_content):
        return "revise", "correction_phrase"
    if candidate_content.strip() == matched_content.strip():
        return "ignore", "no_new_information"
    return "reinforce", "confirmed_existing_memory"
