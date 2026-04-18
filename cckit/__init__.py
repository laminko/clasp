"""cckit — high-level async framework for the Claude Code CLI."""

from .agents import (
    BaseAgent,
    CodeAgent,
    ConversationAgent,
    CustomAgent,
    ResearchAgent,
)
from .core import ACPConfig, CLIConfig, CLI, CommandBuilder, SessionConfig
from .mcp import MCPManager, MCPServer
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
    "CLIConfig",
    "CLI",
    "CommandBuilder",
    "SessionConfig",
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
