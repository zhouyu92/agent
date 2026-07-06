from agent_app.memory_evolution import choose_evolution_action, looks_like_correction


def test_choose_evolution_action_revise_for_correction_phrase():
    action, reason = choose_evolution_action(
        "以后回答不是先铺垫，而是先给结论。",
        "用户喜欢先铺垫。",
    )
    assert action == "revise"
    assert reason == "correction_phrase"


def test_choose_evolution_action_ignore_for_identical_content():
    action, reason = choose_evolution_action("用户喜欢先给结论。", "用户喜欢先给结论。")
    assert action == "ignore"
    assert reason == "no_new_information"


def test_choose_evolution_action_reinforce_for_related_but_different():
    action, reason = choose_evolution_action("用户还是喜欢先给结论。", "用户喜欢先给结论。")
    assert action == "reinforce"
    assert reason == "confirmed_existing_memory"


def test_looks_like_correction_detects_markers():
    assert looks_like_correction("以后回答改成先给结论")
    assert looks_like_correction("实际上不是这样")
    assert not looks_like_correction("用户喜欢简洁的回复")
