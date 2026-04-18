#!/usr/bin/env python3
"""Mock ACP server that simulates the Claude CLI's stream-json protocol.

Implements the essential ACP JSON-RPC methods:
- initialize → returns server capabilities
- session/new → creates a session, returns sessionId
- session/load → loads an existing session
- session/prompt → sends back session/update notifications then a result
- session/close → closes the session
- session/cancel → no-op

Also exercises agent→client callbacks:
- Sends a session/request_permission request for tool use
- Sends session/update notifications (content_delta, result)
"""

import json
import sys


def send_response(msg_id, result=None, error=None):
    resp = {"jsonrpc": "2.0", "id": msg_id}
    if error:
        resp["error"] = error
    else:
        resp["result"] = result
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def send_notification(method, params):
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def send_request(req_id, method, params):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    session_id = ""
    callback_id = 5000

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = data.get("id")
        method = data.get("method")

        # Response to our callback (no method, has id) — skip
        if method is None and msg_id is not None:
            continue

        if method == "initialize":
            send_response(
                msg_id,
                {
                    "capabilities": {"streaming": True, "sessions": True},
                    "serverInfo": {"name": "mock-claude", "version": "0.0.1"},
                },
            )

        elif method == "session/new":
            session_id = "mock-session-001"
            send_response(msg_id, {"sessionId": session_id})

        elif method == "session/load":
            session_id = data.get("params", {}).get(
                "sessionId", "loaded-session"
            )
            send_response(msg_id, {"sessionId": session_id, "loaded": True})

        elif method == "session/prompt":
            message = data.get("params", {}).get("message", "")

            # First, request permission (agent→client callback)
            callback_id += 1
            send_request(
                callback_id,
                "session/request_permission",
                {
                    "tool_name": "TextEditor",
                    "description": "Generating response",
                },
            )
            # Read the permission response
            perm_line = sys.stdin.readline().strip()
            # We don't need to act on the response for the mock

            # Send streaming content_delta notifications
            send_notification(
                "session/update",
                {
                    "type": "assistant_item_started",
                },
            )

            # Simulate token-by-token streaming
            if "2+2" in message:
                chunks = ["4"]
            elif "favorite color" in message.lower():
                chunks = ["blue"]
            elif "count" in message.lower():
                chunks = ["1\n", "2\n", "3\n"]
            else:
                chunks = ["Hello", " from", " mock", " ACP", " server!"]

            for chunk in chunks:
                send_notification(
                    "session/update",
                    {
                        "type": "content_delta",
                        "delta": {"text": chunk},
                    },
                )

            full_text = "".join(chunks)

            send_notification(
                "session/update",
                {
                    "type": "assistant_item_completed",
                    "stop_reason": "end_turn",
                    "session_id": session_id,
                },
            )

            # Send result notification to signal completion
            send_notification(
                "session/update",
                {
                    "type": "result",
                    "result": full_text,
                    "session_id": session_id,
                    "duration_ms": 42,
                    "is_error": False,
                },
            )

            # Respond to the prompt request itself
            send_response(msg_id, {"ok": True})

        elif method == "session/close":
            session_id = ""
            send_response(msg_id, {"closed": True})

        elif method == "session/cancel":
            send_response(msg_id, {"cancelled": True})

        elif method == "exit":
            send_response(msg_id, {"exiting": True})
            sys.exit(0)

        else:
            send_response(
                msg_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            )


if __name__ == "__main__":
    main()
