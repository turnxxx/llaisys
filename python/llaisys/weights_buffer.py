from ctypes import c_char_p

from .libllaisys import LIB_LLAISYS, llaisysWeightBuffer_t
from .tensor import Tensor


class WeightBuffer:
    def __init__(self):
        self._buffer: llaisysWeightBuffer_t = LIB_LLAISYS.weightBufferCreate()

    def __del__(self):
        if hasattr(self, "_buffer") and self._buffer:
            LIB_LLAISYS.weightBufferDestroy(self._buffer)
            self._buffer = None

    def add(self, name: str, weight: Tensor):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        LIB_LLAISYS.weightBufferAdd(
            self._buffer, c_char_p(name.encode("utf-8")), weight.lib_tensor()
        )

    def has(self, name: str) -> bool:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        return bool(LIB_LLAISYS.weightBufferHas(self._buffer, c_char_p(name.encode("utf-8"))))

    def size(self) -> int:
        return int(LIB_LLAISYS.weightBufferSize(self._buffer))

    def clear(self):
        LIB_LLAISYS.weightBufferClear(self._buffer)

    def handle(self) -> llaisysWeightBuffer_t:
        return self._buffer

