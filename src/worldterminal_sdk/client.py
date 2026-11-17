"""Synchronous client for the WorldTerminal hosted policy API."""

import os
import threading
import uuid
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from ._codec import pack, unpack
from .embodiment import JointProfile, check_vectors
from .types import ActionBatch, Observation


class APIError(RuntimeError):
    """The server rejected a request."""


class ConnectionLostError(RuntimeError):
    """Request outcome is uncertain. No action request is automatically replayed."""


ConnectionLost = ConnectionLostError


class Client:
    """One session, one ordered control loop. Use one Client per robot.

    Endpoint/auth default to WORLDTERMINAL_API_URL / WORLDTERMINAL_API_TOKEN.
    Connects on construction; close or use a context manager to release resources.
    """

    def __init__(self, url=None, *, api_key=None, embodiment=None, timeout=120.0, connect_timeout=30.0):
        url = url or os.environ.get("WORLDTERMINAL_API_URL", "")
        parsed = urlsplit(url)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError("set WORLDTERMINAL_API_URL or pass a ws:// or wss:// URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "pass credentials with api_key; URL credentials/query/fragment are not supported"
            )
        if timeout <= 0 or connect_timeout <= 0:
            raise ValueError("timeouts must be positive")
        token = api_key if api_key is not None else os.environ.get("WORLDTERMINAL_API_TOKEN")
        self._embodiment = None if embodiment is None else JointProfile(embodiment)
        self.timeout = timeout
        self.session_id = uuid.uuid4().hex
        self.trial_id = None
        self._step = 0
        self._lock = threading.RLock()
        self._connection = connect(
            url,
            additional_headers={"Authorization": f"Bearer {token}"} if token else None,
            open_timeout=connect_timeout,
            close_timeout=5,
            max_size=64 * 1024 * 1024,
            compression=None,
            proxy=None,
        )
        self._socket = self._connection.__enter__()
        try:
            self.server_info = self._request("hello", {}, "hello_ack")
            if self.server_info.get("worldterminal_protocol") != 1:
                raise APIError("This SDK requires WorldTerminal protocol 1 on the server")
            if self._embodiment and self._embodiment.id not in self.server_info.get("embodiments", []):
                raise APIError(f"Server does not advertise {self._embodiment.id}")
        except BaseException:
            if self._socket is not None:
                self._socket.close()
            self._socket = None
            raise

    def _frame(self, kind, payload):
        return {
            "message_type": kind,
            "message_id": uuid.uuid4().hex,
            "evaluation_id": self.session_id,
            "trial_id": self.trial_id,
            "step": self._step,
            "payload": payload,
        }

    def _request(self, kind, payload, expected):
        with self._lock:
            if self._socket is None:
                raise ConnectionLost("client is closed; create a new session")
            frame = self._frame(kind, payload)
            data = pack(frame)  # serialization errors do not change the connection
            try:
                self._socket.send(data)
                reply = unpack(self._socket.recv(timeout=self.timeout))
                if (
                    reply.get("message_id") != frame["message_id"]
                    or reply.get("evaluation_id") != self.session_id
                ):
                    raise ValueError("API response does not match the request")
                if reply.get("message_type") == "error":
                    raise APIError(
                        str(reply.get("payload", {}).get("message", "API request failed"))
                    )
                if reply.get("message_type") != expected:
                    raise ValueError("unexpected API response type")
                return reply.get("payload", {})
            except APIError:
                raise
            except Exception as error:
                self._socket.close()
                self._socket = None
                raise ConnectionLost("API connection failed; request was not replayed") from error

    def reset(self):
        """Start a fresh episode in this session."""
        with self._lock:
            self.trial_id = uuid.uuid4().hex
            self._step = 0
            return self._request("reset", {"trial_id": self.trial_id}, "reset_result").get("result")

    def _call(self, method, observation=None):
        if self.trial_id is None:
            raise RuntimeError("call reset() before starting an episode")
        payload = {"func_name": method}
        if observation is not None:
            payload["obs"] = observation
        return self._request("call", payload, "call_result").get("result")

    def update_observation(self, observation):
        """Send a fresh observation without polling for an action."""
        data = observation.to_dict() if isinstance(observation, Observation) else dict(observation)
        if self._embodiment:
            data = self._embodiment.observation(data)
        return self._call("update_obs", data)

    def get_action(self):
        """Return a validated {status, actions} result for an adapter control loop."""
        with self._lock:
            result = self._call("get_action")
            self._step += 1
            batch = ActionBatch.from_wire(result)
            if self._embodiment:
                for index, row in enumerate(batch.actions):
                    check_vectors(row, self._embodiment.action, f"action[{index}]")
            return result

    def step(self, observation):
        """Send one observation and return actions, waiting, or done.

        If waiting, call poll() without resending the same observation.
        """
        with self._lock:
            self.update_observation(observation)
            return self.poll()

    def poll(self):
        """Wait for the next action batch without resending an observation."""
        return ActionBatch.from_wire(self.get_action())

    def end(self, result=None):
        """Report the caller's outcome; done alone must not be reported as success."""
        with self._lock:
            if self.trial_id is None:
                raise RuntimeError("no active episode")
            payload = dict(result or {})
            payload["trial_id"] = self.trial_id
            response = self._request("trial_end", payload, "trial_end_ack").get("result")
            self.trial_id = None
            return response

    def close(self):
        """Release the hosted session and local connection."""
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.send(pack(self._frame("close", {})))
                except ConnectionClosed:
                    pass  # Closing an already disconnected client must be idempotent.
                finally:
                    self._socket.close()
                    self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
