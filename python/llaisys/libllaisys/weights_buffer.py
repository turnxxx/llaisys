import ctypes
from ctypes import c_char_p, c_size_t, c_uint8
from .tensor import llaisysTensor_t

llaisysWeightBuffer_t = ctypes.c_void_p


def load_weights_buffer(lib):
    lib.weightBufferCreate.argtypes = []
    lib.weightBufferCreate.restype = llaisysWeightBuffer_t

    lib.weightBufferDestroy.argtypes = [llaisysWeightBuffer_t]
    lib.weightBufferDestroy.restype = None

    lib.weightBufferClear.argtypes = [llaisysWeightBuffer_t]
    lib.weightBufferClear.restype = None

    lib.weightBufferSize.argtypes = [llaisysWeightBuffer_t]
    lib.weightBufferSize.restype = c_size_t

    lib.weightBufferHas.argtypes = [llaisysWeightBuffer_t, c_char_p]
    lib.weightBufferHas.restype = c_uint8

    lib.weightBufferAdd.argtypes = [llaisysWeightBuffer_t, c_char_p, llaisysTensor_t]
    lib.weightBufferAdd.restype = None

