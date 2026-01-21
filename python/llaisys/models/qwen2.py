from typing import Sequence
from pathlib import Path
import ctypes

import numpy as np
import safetensors

from ..libllaisys import DeviceType, DataType, MemcpyKind
from ..runtime import RuntimeAPI
from ..tensor import Tensor


def _llaisys_dtype_from_numpy(np_dtype: np.dtype) -> DataType:
    if np_dtype == np.float16:
        return DataType.F16
    if np_dtype == np.float32:
        return DataType.F32
    if np_dtype == np.float64:
        return DataType.F64
    if str(np_dtype) == "bfloat16":
        return DataType.BF16
    if np_dtype == np.int8:
        return DataType.I8
    if np_dtype == np.int16:
        return DataType.I16
    if np_dtype == np.int32:
        return DataType.I32
    if np_dtype == np.int64:
        return DataType.I64
    if np_dtype == np.uint8:
        return DataType.U8
    if np_dtype == np.uint16:
        return DataType.U16
    if np_dtype == np.uint32:
        return DataType.U32
    if np_dtype == np.uint64:
        return DataType.U64
    if np_dtype == np.bool_:
        return DataType.BOOL
    raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        model_path = Path(model_path)
        self.device = device
        self.device_id = 0
        self.weights = {}

        api = RuntimeAPI(device)
        memcpy_kind = MemcpyKind.H2H if device == DeviceType.CPU else MemcpyKind.H2D

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    np_tensor = data_.get_tensor(name_)
                    np_tensor = np.ascontiguousarray(np_tensor)
                    llaisys_dtype = _llaisys_dtype_from_numpy(np_tensor.dtype)
                    llaisys_tensor = Tensor(
                        np_tensor.shape,
                        dtype=llaisys_dtype,
                        device=device,
                        device_id=self.device_id,
                    )
                    api.memcpy_sync(
                        llaisys_tensor.data_ptr(),
                        ctypes.c_void_p(np_tensor.ctypes.data),
                        np_tensor.nbytes,
                        memcpy_kind,
                    )
                    self.weights[name_] = llaisys_tensor

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        return []
