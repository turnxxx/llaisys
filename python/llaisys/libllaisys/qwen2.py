import ctypes
from ctypes import c_int64, c_size_t, c_float, c_int, POINTER

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .weights_buffer import llaisysWeightBuffer_t


llaisysQwen2Model_t = ctypes.c_void_p


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
        ("attention_dropout", c_float),
        ("initializer_range", c_float),
        ("max_window_layers", c_size_t),
        ("sliding_window", c_size_t),
        ("tie_word_embeddings", c_int),
        ("use_cache", c_int),
        ("use_mrope", c_int),
        ("use_sliding_window", c_int),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelLoadWeights.argtypes = [llaisysQwen2Model_t, llaisysWeightBuffer_t]
    lib.llaisysQwen2ModelLoadWeights.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = ctypes.c_void_p

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferDialog.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
        c_size_t,
        POINTER(c_int64),
        c_size_t,
    ]
    lib.llaisysQwen2ModelInferDialog.restype = c_int64


