import json
import warnings

from abc import ABC, abstractmethod
from io import BytesIO

import msgpack
import numpy as np
import torch

try:
    import zmq
except ImportError:
    warnings.warn("zmq is not installed, please install it with `pip install pyzmq` for full functionality of ZMQServicePolicy", ImportWarning)


class Policy(ABC):
    def __init__(self, type: str):
        self.type = type

    @abstractmethod
    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """
        Get the action from the policy.
        Expect the action to be a torch.Tensor with shape (action_horizon, env_num, action_dim).
        """
        pass


class CheckpointPolicy(Policy):
    def __init__(self, checkpoint_path: str):
        super().__init__("checkpoint")
        self.checkpoint_path = checkpoint_path

    def get_action(self, *args, **kwargs) -> torch.Tensor:
        pass


class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class MsgPackSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgPackSerializer._encode_custom)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgPackSerializer._decode_custom)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, np.ndarray):
            buffer = BytesIO()
            np.save(buffer, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        return obj

    @staticmethod
    def _decode_custom(obj):
        if "__ndarray_class__" in obj:
            return np.load(BytesIO(obj["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in obj:
            return json.loads(obj["as_json"])
        return obj


class ZMQServicePolicy(Policy):
    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        ping_endpoint: str = "ping",
        serializer=TorchSerializer,
    ):
        super().__init__("service")
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()
        self._ping_endpoint = ping_endpoint
        self._serializer = serializer
        self.check_service_status()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def check_service_status(self):
        if not self._ping():
            raise RuntimeError("Service is not running, please start the service first.")
        else:
            print("Service is running.")

    def _ping(self) -> bool:
        if self._ping_endpoint is None:
            raise ValueError("ping_endpoint is not set")
        try:
            self.call_endpoint(self._ping_endpoint, requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(self._serializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return self._serializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()
