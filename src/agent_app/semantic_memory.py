from __future__ import annotations

import re

TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "结论": ("重点", "要点"),
    "回答": ("回复", "答复"),
    "偏好": ("喜欢", "喜好"),
}

MEMORY_CATEGORIES = frozenset({"fact", "preference", "relationship", "feedback", "principle", "strategy", "general"})


def normalize_memory_category(category: str) -> str:
    normalized = category.strip().lower()
    return normalized if normalized in MEMORY_CATEGORIES else "general"


def query_terms(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    chars = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
    terms = words | cjk | chars
    for canonical, aliases in TERM_ALIASES.items():
        if canonical in text or any(alias in text for alias in aliases):
            terms.add(canonical)
    return terms


def memory_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    terms |= set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    keywords = [
        "结论",
        "原因",
        "偏好",
        "回答",
        "结构",
        "直接",
        "详细",
        "推演",
        "温度",
        "风格",
        "边界",
        "原则",
    ]
    terms |= {keyword for keyword in keywords if keyword in text}
    for canonical, aliases in TERM_ALIASES.items():
        if canonical in text or any(alias in text for alias in aliases):
            terms.add(canonical)
    return terms


def category_alias(category: str) -> str:
    aliases = {
        "preference": "偏好 喜好",
        "fact": "事实 信息",
        "relationship": "关系",
        "feedback": "反馈 评价",
        "principle": "原则 规则",
        "strategy": "策略 方法 互动",
        "general": "通用",
    }
    return aliases.get(category.lower(), "")


def is_similar_memory(existing_contents: list[str], candidate_content: str) -> bool:
    new_terms = memory_terms(candidate_content)
    if len(new_terms) < 3:
        return False
    for existing_content in existing_contents:
        existing_terms = memory_terms(existing_content)
        if len(existing_terms) < 3:
            continue
        overlap = len(new_terms & existing_terms)
        ratio = overlap / min(len(new_terms), len(existing_terms))
        if overlap >= 3 and ratio >= 0.65:
            return True
    return False


def score_memory_match(query: str, content: str, category: str, importance: int) -> int:
    overlap = len(query_terms(query) & query_terms(content + " " + category + " " + category_alias(category)))
    if overlap == 0:
        return 0
    return overlap * 10 + importance
