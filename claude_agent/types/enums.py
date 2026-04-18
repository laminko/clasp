from enum import Enum


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class PermissionMode(Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    DONT_ASK = "dontAsk"
    BYPASS = "bypassPermissions"
    PLAN = "plan"
