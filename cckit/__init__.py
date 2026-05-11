"""cckit — high-level async framework for the Claude Code CLI."""

from .agents import (
    BaseAgent,
    CodeAgent,
    ConversationAgent,
    CustomAgent,
    ResearchAgent,
)
from .core import (
    ACPConfig,
    CLAUDE_BARE_ALIASES,
    CLIConfig,
    CLI,
    CommandBuilder,
    SessionConfig,
    discover_claude_models,
)
from .mcp import FastMCP, MCPManager, MCPServer
from .rpc import ACPClient, PermissionPolicy, RpcTransport
from .session import ACPSession, ConversationManager, MessageHistory, Session
from .streaming import (
    Event,
    MessageCompleteEvent,
    ResultEvent,
    StreamHandler,
    TextChunkEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from .types import (
    AgentResult,
    InputFormat,
    Message,
    OutputFormat,
    PermissionMode,
    Response,
    Usage,
)
from .utils import (
    AuthError,
    CLIError,
    CckitError,
    ParseError,
    ProtocolError,
    RpcError,
    SessionError,
    TimeoutError,
    TransportError,
)

__all__ = [
    # core
    "ACPConfig",
    "CLAUDE_BARE_ALIASES",
    "CLIConfig",
    "CLI",
    "CommandBuilder",
    "SessionConfig",
    "discover_claude_models",
    # session
    "ACPSession",
    "ConversationManager",
    "MessageHistory",
    "Session",
    # rpc
    "ACPClient",
    "PermissionPolicy",
    "RpcTransport",
    # agents
    "BaseAgent",
    "CodeAgent",
    "ConversationAgent",
    "CustomAgent",
    "ResearchAgent",
    # mcp
    "FastMCP",
    "MCPManager",
    "MCPServer",
    # streaming
    "Event",
    "MessageCompleteEvent",
    "ResultEvent",
    "StreamHandler",
    "TextChunkEvent",
    "ToolResultEvent",
    "ToolUseEvent",
    # types
    "AgentResult",
    "InputFormat",
    "Message",
    "OutputFormat",
    "PermissionMode",
    "Response",
    "Usage",
    # errors
    "AuthError",
    "CLIError",
    "CckitError",
    "ParseError",
    "ProtocolError",
    "RpcError",
    "SessionError",
    "TimeoutError",
    "TransportError",
]
