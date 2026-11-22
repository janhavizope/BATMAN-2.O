"""
node_rpc.py
A minimal JSON-RPC client for a local Bitcoin Core node (bitcoind). Only the
standard library is used: HTTP POST with basic auth, reading the node's
`.cookie` file when no user/password is configured.

This is how the ZMQ feed turns a transaction id into full details
(getrawtransaction with verbosity 2 returns each input's previous output,
so we get the sending addresses and amounts). It talks only to the local
node — nothing leaves the machine.

Connection settings come from the environment, with sensible localhost
defaults (see src/config.py):
    BATMAN_RPC_URL      default http://127.0.0.1:8332
    BATMAN_RPC_COOKIE   path to bitcoind's .cookie file (preferred)
    BATMAN_RPC_USER / BATMAN_RPC_PASSWORD   if you set rpcuser/rpcpassword
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


class NodeRPCError(RuntimeError):
    """Raised when the node cannot be reached or returns an RPC error."""


class BitcoinRPC:
    def __init__(self, url: str, user: Optional[str] = None, password: Optional[str] = None,
                 cookie_path: Optional[str] = None, timeout: float = 15.0):
        self.url = url
        self.timeout = timeout
        self._auth = self._resolve_auth(user, password, cookie_path)

    @staticmethod
    def _resolve_auth(user, password, cookie_path) -> str:
        """Return the basic-auth header value, from user/pass or the cookie file."""
        if user and password is not None:
            token = f"{user}:{password}"
        elif cookie_path:
            cookie = Path(cookie_path)
            if not cookie.exists():
                raise NodeRPCError(f"RPC cookie file not found: {cookie_path}")
            token = cookie.read_text(encoding="utf-8").strip()
        else:
            raise NodeRPCError("no RPC credentials: set a cookie path or user/password")
        return "Basic " + base64.b64encode(token.encode("utf-8")).decode("ascii")

    def call(self, method: str, params: Optional[list] = None) -> Any:
        payload = json.dumps({"jsonrpc": "1.0", "id": "batman", "method": method,
                              "params": params or []}).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload,
            headers={"Authorization": self._auth, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # bitcoind returns the RPC error as the body even on HTTP 500.
            detail = ""
            try:
                detail = json.loads(error.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:
                pass
            raise NodeRPCError(f"{method}: HTTP {error.code} {detail}".strip()) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise NodeRPCError(f"could not reach the node at {self.url}: {error}") from error
        if body.get("error"):
            raise NodeRPCError(f"{method}: {body['error']}")
        return body["result"]

    def get_transaction_verbose(self, txid: str) -> dict:
        """getrawtransaction with verbosity 2: includes each input's prevout."""
        return self.call("getrawtransaction", [txid, 2])

    def ping(self) -> bool:
        """True if the node answers. Raises NodeRPCError if it cannot be reached."""
        self.call("getblockchaininfo")
        return True
