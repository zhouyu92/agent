from agent_app.semantic_memory import is_similar_memory, score_memory_match


def test_is_similar_memory_detects_high_overlap_preference():
    existing_contents = ["用户要求交流时先给出结论，再补充原因。"]

    assert is_similar_memory(existing_contents, "用户偏好回答结构：先给结论，再补充原因。") is True


def test_is_similar_memory_allows_distinct_content():
    existing_contents = ["用户要求交流时先给出结论，再补充原因。"]

    assert is_similar_memory(existing_contents, "用户正在搭建一个能自我学习的 agent。") is False


def test_score_memory_match_uses_category_alias_and_importance():
    better = score_memory_match("回答偏好", "用户喜欢先给结论。", "preference", 4)
    worse = score_memory_match("回答偏好", "用户正在搭建一个 agent。", "fact", 4)

    assert better > worse
    assert better > 0
