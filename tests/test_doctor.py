import os
import subprocess
import sys

from agent_app.config import AgentConfig
from agent_app.doctor import DoctorResult, run_doctor


class FakeProbeClient:
    def __init__(self):
        self.called = False

    def ping(self):
        self.called = True
        return "ok"


def test_doctor_offline_checks_config_without_calling_model(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        user_id="alice",
    )
    client = FakeProbeClient()

    result = run_doctor(config, client=client, online=False)

    assert result == DoctorResult(ok=True, checks=["config", "memory"], errors=[])
    assert client.called is False


def test_doctor_online_pings_model(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    client = FakeProbeClient()

    result = run_doctor(config, client=client, online=True)

    assert result.ok is True
    assert "model" in result.checks
    assert client.called is True


def test_doctor_module_entrypoint_prints_offline_checks(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["DASHSCOPE_API_KEY"] = "test-key"
    env["DASHSCOPE_WORKSPACE_ID"] = "workspace"
    env["AGENT_MEMORY_DB"] = str(tmp_path / "agent.db")

    result = subprocess.run(
        [sys.executable, "-m", "agent_app.doctor"],
        cwd=".",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ok: config" in result.stdout
    assert "ok: memory" in result.stdout
