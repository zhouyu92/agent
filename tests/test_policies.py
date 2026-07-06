from agent_app.policies import TurnRoutingPolicy


def test_turn_routing_policy_retrieves_but_does_not_learn_on_recall_turn():
    policy = TurnRoutingPolicy()
    decision = policy.evaluate("你还记得我的回答偏好吗？")

    assert decision.should_retrieve is True
    assert decision.retrieve_reason == "default_retrieve"
    assert decision.should_learn is False
    assert decision.learn_reason == "recall_turn"


def test_turn_routing_policy_skips_both_for_low_signal_turn():
    policy = TurnRoutingPolicy()
    decision = policy.evaluate("谢谢")

    assert decision.should_retrieve is False
    assert decision.retrieve_reason == "low_signal"
    assert decision.should_learn is False
    assert decision.learn_reason == "low_signal"
