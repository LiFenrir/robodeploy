"""TCP + pickle protocol helpers for communicating with LeRobot inference server.

Mirrors the protocol used by pi05_server.py: 4-byte big-endian length header
followed by pickle-serialized payload.
"""

import pickle
import socket
import struct


def send_msg(sock: socket.socket, obj: object) -> None:
    """Send a pickle-serialized object over a TCP socket."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a TCP socket."""
    chunks = []
    received = 0
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            raise ConnectionError("socket connection closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> object:
    """Receive a pickle-serialized object from a TCP socket."""
    header = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", header)[0]
    data = recv_exact(sock, msg_len)
    return pickle.loads(data)
