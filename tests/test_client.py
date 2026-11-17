"""Independent SDK protocol and failure handling tests."""

import threading
from contextlib import contextmanager

import msgpack
import numpy as np
import pytest
from websockets.sync.server import serve
from worldterminal_sdk import Client, ConnectionLost, Observation
from worldterminal_sdk._codec import pack, unpack


@contextmanager
def endpoint(*, drop=False, wrong_id=False, close_after_reset=False):
    """Serve a controllable local endpoint for failure and roundtrip tests."""
    seen = []

    def handler(ws):
        for wire in ws:
            frame = unpack(wire)
            seen.append(frame)
            kind, payload = frame["message_type"], frame["payload"]
            if kind == "close":
                return
            if drop and kind == "call":
                ws.close()
                return
            response = {"worldterminal_protocol": 1} if kind == "hello" else {}
            if kind == "call" and payload["func_name"] == "get_action":
                response["result"] = {"status": "ready", "actions": [{"target_position": [1.0, 2.0]}]}
            result = {
                **frame,
                "message_type": {
                    "hello": "hello_ack",
                    "reset": "reset_result",
                    "call": "call_result",
                    "trial_end": "trial_end_ack",
                }[kind],
                "payload": response,
            }
            if wrong_id:
                result["message_id"] = "other-request"
            ws.send(pack(result))
            if close_after_reset and kind == "reset":
                ws.close()
                return

    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", seen
    finally:
        server.shutdown()
        thread.join(5)


def test_unrelated_robot_schema_and_lossless_camera_roundtrip():
    """Preserve arbitrary robot fields and image bytes across two episodes."""
    with endpoint() as (url, seen), Client(url) as client:
        for _ in range(2):
            client.reset()
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            result = client.step(
                Observation(
                    state={"position": np.array([0.0, 0.0])},
                    vision={"overhead": {"color": rgb}},
                    instruction="Move to the marker",
                )
            )
            assert result.actions == ({"target_position": [1.0, 2.0]},)
            sent = [f for f in seen if f["payload"].get("func_name") == "update_obs"][-1]
            np.testing.assert_array_equal(
                sent["payload"]["obs"]["vision"]["overhead"]["color"], rgb
            )
            client.end({"success": True})
        ids = [f["trial_id"] for f in seen if f["message_type"] == "reset"]
        assert len(set(ids)) == 2


def test_dropped_action_request_is_not_replayed():
    """Leave an ambiguous action request unreplayed after a disconnect."""
    with endpoint(drop=True) as (url, seen), Client(url) as client:
        client.reset()
        with pytest.raises(ConnectionLost):
            client.get_action()
        assert len([f for f in seen if f["message_type"] == "call"]) == 1
        with pytest.raises(ConnectionLost):
            client.get_action()


def test_mismatched_response_rejected():
    """Reject a response addressed to a different request."""
    with endpoint(wrong_id=True) as (url, _), pytest.raises(ConnectionLost):
        Client(url)


def test_object_array_and_pickle_wire_rejected():
    """Reject object arrays before decoding their contents."""
    with pytest.raises(TypeError):
        pack({"data": np.array([object()], dtype=object)})
    with pytest.raises(ValueError):
        unpack(msgpack.packb({b"nd": True, b"type": "|O", b"data": b"not-pickle", b"shape": [1]}))


def test_wait_is_not_done_and_empty_actions_do_not_claim_success():
    """Distinguish waiting from termination without claiming task success."""
    from worldterminal_sdk import ActionBatch

    assert ActionBatch.from_wire({"status":"waiting","actions":[]}).waiting
    assert not ActionBatch.from_wire({"status":"waiting","actions":[]}).done
    assert ActionBatch.from_wire({"status":"done","actions":[]}).done


def test_closed_connection_does_not_mask_caller_error():
    """Preserve an application failure when cleanup finds a disconnected socket."""
    from websockets.exceptions import ConnectionClosed

    with (
        endpoint(close_after_reset=True) as (url, _),
        pytest.raises(ValueError, match="caller error"),
        Client(url) as client,
    ):
        client.reset()
        with pytest.raises(ConnectionClosed):
            client._socket.recv(timeout=1)
        raise ValueError("caller error")


@pytest.mark.parametrize("bad", [[], {}, {"status": "ready", "actions": []},
    {"status": "done", "actions": [{}]}, {"status": "waiting", "actions": [{}]},
    {"status": "other", "actions": []}, {"status": "ready", "actions": [1]}])
def test_invalid_action_response(bad):
    from worldterminal_sdk import ActionBatch

    with pytest.raises(ValueError):
        ActionBatch.from_wire(bad)
