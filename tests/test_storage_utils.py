from agent_app.storage_utils import clean_user_id, looks_sensitive, redact_sensitive


def test_clean_user_id_defaults_blank_values():
    assert clean_user_id("") == "default"
    assert clean_user_id("   ") == "default"
    assert clean_user_id("alice") == "alice"


def test_looks_sensitive_detects_api_key_patterns():
    assert looks_sensitive("我的 key 是 sk-1234567890abcdef1234567890abcdef") is True
    assert looks_sensitive("这只是普通对话内容") is False


def test_redact_sensitive_masks_api_keys():
    text = "我的 key 是 sk-1234567890abcdef1234567890abcdef"

    assert "sk-1234567890abcdef1234567890abcdef" not in redact_sensitive(text)
    assert "[REDACTED_SECRET]" in redact_sensitive(text)
