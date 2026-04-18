#!/usr/bin/env python3
"""Minimal JSON-RPC echo server over stdin/stdout for transport tests.

Reads NDJSON from stdin, responds on stdout. Supports:
- "initialize" → returns {"capabilities": {}}
- "echo" → returns the params back
- "error" → returns a JSON-RPC error
- "slow" → sleeps before responding (for timeout tests)
- "notify_back" → sends a notification back to the client
- "callback" → sends a request back to the client, returns client's response
"""

import json
import sys
import time


def respond(msg_id, result=None, error=None):
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
    callback_id = 9000

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

        # If it's a response to our callback request (no method, has id)
        if method is None and msg_id is not None:
            # This is a response to our callback — ignore for now
            continue

        if method == "initialize":
            respond(msg_id, {"capabilities": {}})

        elif method == "echo":
            respond(msg_id, data.get("params", {}))

        elif method == "error":
            respond(
                msg_id,
                error={
                    "code": -32000,
                    "message": "Test error",
                    "data": "details",
                },
            )

        elif method == "slow":
            delay = data.get("params", {}).get("delay", 5)
            time.sleep(delay)
            respond(msg_id, {"delayed": True})

        elif method == "notify_back":
            # Send a notification back to the client
            send_notification("test/notification", data.get("params", {}))
            respond(msg_id, {"notified": True})

        elif method == "callback":
            # Send a request back to the client and wait for response
            callback_id += 1
            send_request(callback_id, "test/callback", data.get("params", {}))
            # Read the response from client
            resp_line = sys.stdin.readline().strip()
            if resp_line:
                resp_data = json.loads(resp_line)
                respond(msg_id, {"callback_result": resp_data.get("result")})
            else:
                respond(
                    msg_id,
                    error={"code": -32603, "message": "No callback response"},
                )

        elif method == "exit":
            respond(msg_id, {"exiting": True})
            sys.exit(0)

        else:
            respond(
                msg_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            )


if __name__ == "__main__":
    main()
