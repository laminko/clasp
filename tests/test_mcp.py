"""Tests for ``cckit.mcp`` helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cckit import FastMCP, MCPManager
from cckit.mcp.server import MCPServer


class TestAddPythonServer:
    def test_script_path(self, tmp_path: Path) -> None:
        script = tmp_path / "srv.py"
        script.write_text("# stub\n")

        mgr = MCPManager()
        mgr.add_python_server("demo", script=script)

        cfg = mgr.to_config()["mcpServers"]["demo"]
        assert cfg["command"] == sys.executable
        assert cfg["args"] == [str(script.resolve())]

    def test_module_form(self) -> None:
        mgr = MCPManager()
        mgr.add_python_server("demo", module="my_pkg.server")

        cfg = mgr.to_config()["mcpServers"]["demo"]
        assert cfg["command"] == sys.executable
        assert cfg["args"] == ["-m", "my_pkg.server"]

    def test_custom_python_and_extra_args(self, tmp_path: Path) -> None:
        script = tmp_path / "srv.py"
        script.write_text("")

        mgr = MCPManager()
        mgr.add_python_server(
            "demo",
            script=script,
            python="/opt/venv/bin/python",
            args=["--flag", "value"],
            env={"LOG_LEVEL": "debug"},
        )

        cfg = mgr.to_config()["mcpServers"]["demo"]
        assert cfg["command"] == "/opt/venv/bin/python"
        assert cfg["args"] == [str(script.resolve()), "--flag", "value"]
        assert cfg["env"] == {"LOG_LEVEL": "debug"}

    def test_requires_exactly_one_of_script_or_module(self) -> None:
        mgr = MCPManager()

        with pytest.raises(ValueError, match="exactly one"):
            mgr.add_python_server("demo")

        with pytest.raises(ValueError, match="exactly one"):
            mgr.add_python_server("demo", script="a.py", module="m")

    def test_config_file_roundtrip(self, tmp_path: Path) -> None:
        script = tmp_path / "srv.py"
        script.write_text("")

        mgr = MCPManager()
        mgr.add_python_server("demo", script=script)
        path = mgr.write_config_file(tmp_path / "mcp.json")

        loaded = json.loads(Path(path).read_text())
        assert "mcpServers" in loaded
        assert loaded["mcpServers"]["demo"]["command"] == sys.executable


class TestMCPServer:
    def test_to_dict_minimal(self) -> None:
        s = MCPServer(name="fs", command="node")
        d = s.to_dict()
        assert d == {"command": "node", "args": []}

    def test_to_dict_with_env(self) -> None:
        s = MCPServer(
            name="fs",
            command="node",
            args=["server.js"],
            env={"DEBUG": "1"},
        )
        d = s.to_dict()
        assert d["env"] == {"DEBUG": "1"}
        assert d["args"] == ["server.js"]


class TestMCPManagerPlain:
    def test_add_server_returns_self(self) -> None:
        m = MCPManager()
        assert m.add_server("fs", "node", args=["s.js"]) is m

    def test_add_and_remove_server(self) -> None:
        m = MCPManager()
        m.add_server("fs", "node")
        m.add_server("db", "python")
        cfg = m.to_config()["mcpServers"]
        assert set(cfg) == {"fs", "db"}

        m.remove_server("fs")
        m.remove_server("nonexistent")  # noop
        assert set(m.to_config()["mcpServers"]) == {"db"}

    def test_write_config_file_temp_roundtrip(self) -> None:
        m = MCPManager()
        m.add_server("fs", "node")
        path = m.write_config_file()
        try:
            data = json.loads(Path(path).read_text())
            assert data["mcpServers"]["fs"]["command"] == "node"
        finally:
            m.cleanup()
        assert not Path(path).exists()

    def test_cleanup_without_temp_is_noop(self) -> None:
        m = MCPManager()
        m.cleanup()

    def test_repr_lists_servers(self) -> None:
        m = MCPManager()
        m.add_server("fs", "node")
        assert "fs" in repr(m)


class TestFastMCPReexport:
    def test_importable_and_registers_tools(self) -> None:
        server = FastMCP("unit-test")

        @server.tool(description="double a number")
        def double(x: int) -> int:
            return x * 2

        tools = server._tool_manager.list_tools()
        names = {t.name for t in tools}
        assert "double" in names
        tool = next(t for t in tools if t.name == "double")
        assert tool.parameters["properties"]["x"]["type"] == "integer"
