"""Numeric MessagePack wire compatibility, with no pickle/object-array decoding."""

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError("only numeric numpy arrays are supported")
        value = np.ascontiguousarray(value)
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": b"",
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported payload type: {type(value).__name__}")


def _decode(value):
    if b"nd" not in value:
        return value
    dtype = np.dtype(value[b"type"])
    if dtype.kind not in "biufc":
        raise ValueError("only numeric numpy arrays are supported")
    array = np.frombuffer(value[b"data"], dtype=dtype)
    if value[b"nd"]:
        return array.reshape(value[b"shape"]).copy()
    if array.size != 1:
        raise ValueError("invalid numpy scalar")
    return array[0]


def pack(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(data):
    value = msgpack.unpackb(data, raw=False, strict_map_key=False, object_hook=_decode)
    if not isinstance(value, dict):
        raise ValueError("API frame must be a map")
    return value
