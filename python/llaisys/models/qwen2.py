import json
from typing import Sequence, Optional, Dict, Any, List
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta
from ..tensor import Tensor
from ..weights_buffer import WeightBuffer
from pathlib import Path
import safetensors
import re
import torch
import ctypes


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, device_ids: List[int] = None):
        model_path = Path(model_path)
        self._device = device
        self._device_ids = device_ids or []
        self._weight_buffer = WeightBuffer()
        self._model = None
        self._meta = self._load_meta(model_path)
        files = sorted(model_path.glob("*.safetensors"))
        print(f"safetensors files: {len(files)}", flush=True)
        if files:
            print(f"first file: {files[0]}", flush=True)
        for file in files:
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
            for name_ in data_.keys():
              arr = data_.get_tensor(name_)
              llaisys_tensor = self._torch_to_llaisys_tensor(arr, device)
              self._weight_buffer.add(name_, llaisys_tensor)
        print(f"weight buffer size: {self._weight_buffer.size()}", flush=True)

        self._create_model()
        self._load_weights()

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if self._model is None:
            raise RuntimeError("Model is not initialized")
        if max_new_tokens is None:
            max_new_tokens = 1
        if not isinstance(inputs, Sequence) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty sequence of token ids")

        tokens = list(int(t) for t in inputs)
        return self._infer_dialog(tokens, max_new_tokens)

    @staticmethod
    def _parse_weight_name(name: str) -> Optional[Dict[str, Any]]:
        """
        Parse safetensors weight name into a structured mapping.
        This is a template for routing weights into the C++ backend.
        """
        if name == "model.embed_tokens.weight":
            return {"type": "embed_tokens", "layer": None, "param": "weight"}
        if name == "model.norm.weight":
            return {"type": "final_norm", "layer": None, "param": "weight"}
        if name == "lm_head.weight":
            return {"type": "lm_head", "layer": None, "param": "weight"}

        m = re.match(r"^model\.layers\.(\d+)\.(.+)$", name)
        if not m:
            return None
        layer = int(m.group(1))
        suffix = m.group(2)

        # Attention
        if suffix in {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        }:
            return {"type": "attn", "layer": layer, "param": suffix}

        # MLP
        if suffix in {
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }:
            return {"type": "mlp", "layer": layer, "param": suffix}

        # Norms
        if suffix in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }:
            return {"type": "norm", "layer": layer, "param": suffix}

        return {"type": "unknown", "layer": layer, "param": suffix}

    @staticmethod
    def _torch_dtype_to_llaisys(dtype: torch.dtype) -> DataType:
        if dtype == torch.float16:
            return DataType.F16
        if dtype == torch.float32:
            return DataType.F32
        if dtype == torch.float64:
            return DataType.F64
        if dtype == torch.bfloat16:
            return DataType.BF16
        if dtype == torch.int64:
            return DataType.I64
        if dtype == torch.int32:
            return DataType.I32
        if dtype == torch.int16:
            return DataType.I16
        if dtype == torch.int8:
            return DataType.I8
        if dtype == torch.uint8:
            return DataType.U8
        if dtype == torch.bool:
            return DataType.BOOL
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    @staticmethod
    def _torch_to_llaisys_tensor(tensor: torch.Tensor, device: DeviceType) -> Tensor:
        cpu_tensor = tensor.detach().contiguous().cpu()
        llaisys_dtype = Qwen2._torch_dtype_to_llaisys(cpu_tensor.dtype)
        llaisys_tensor = Tensor(shape=cpu_tensor.shape, dtype=llaisys_dtype, device=device)
        if cpu_tensor.dtype == torch.bfloat16:
            # numpy 不支持 bfloat16，直接用 data_ptr 传入原始内存
            llaisys_tensor.load(ctypes.c_void_p(cpu_tensor.data_ptr()))
        else:
            llaisys_tensor.load(cpu_tensor.numpy().ctypes.data_as(ctypes.c_void_p))
        return llaisys_tensor

    @staticmethod
    def _config_dtype_to_llaisys(dtype_name: str) -> DataType:
        name = (dtype_name or "").lower()
        if name in {"float16", "fp16"}:
            return DataType.F16
        if name in {"float32", "fp32"}:
            return DataType.F32
        if name in {"float64", "fp64"}:
            return DataType.F64
        if name in {"bfloat16", "bf16"}:
            return DataType.BF16
        raise ValueError(f"Unsupported dtype in config: {dtype_name}")

    def _load_meta(self, model_path: Path) -> LlaisysQwen2Meta:
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        hs = int(cfg["hidden_size"])
        nh = int(cfg["num_attention_heads"])
        nkvh = int(cfg["num_key_value_heads"])
        meta = LlaisysQwen2Meta()
        meta.dtype = int(self._config_dtype_to_llaisys(cfg.get("torch_dtype", "float32")))
        meta.nlayer = int(cfg["num_hidden_layers"])
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = int(hs // nh)
        meta.di = int(cfg["intermediate_size"])
        meta.maxseq = int(cfg["max_position_embeddings"])
        meta.voc = int(cfg["vocab_size"])
        meta.epsilon = float(cfg["rms_norm_eps"])
        meta.theta = float(cfg["rope_theta"])
        meta.end_token = int(cfg.get("eos_token_id", cfg.get("bos_token_id", 0)))
        meta.attention_dropout = float(cfg.get("attention_dropout", 0.0))
        meta.initializer_range = float(cfg.get("initializer_range", 0.0))
        meta.max_window_layers = int(cfg.get("max_window_layers", 0))
        meta.sliding_window = int(cfg.get("sliding_window", 0))
        meta.tie_word_embeddings = int(bool(cfg.get("tie_word_embeddings", False)))
        meta.use_cache = int(bool(cfg.get("use_cache", False)))
        meta.use_mrope = int(bool(cfg.get("use_mrope", False)))
        meta.use_sliding_window = int(bool(cfg.get("use_sliding_window", False)))
        return meta

    def _create_model(self):
        if self._model:
            return
        device_ids = None
        if self._device_ids:
            device_ids = (ctypes.c_int * len(self._device_ids))(*self._device_ids)
            ndevice = len(self._device_ids)
        else:
            ndevice = 0
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            int(self._device),
            device_ids,
            ndevice,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

    def _load_weights(self):
        if not self._model:
            raise RuntimeError("Model is not initialized")
        LIB_LLAISYS.llaisysQwen2ModelLoadWeights(self._model, self._weight_buffer.handle())

    def _infer_next_token(self, tokens: Sequence[int]) -> int:
        arr = (ctypes.c_int64 * len(tokens))(*tokens)
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(tokens)))

    def _infer_dialog(self, tokens: Sequence[int], max_steps: int) -> List[int]:
        if max_steps is None:
            max_steps = 1
        in_buf = (ctypes.c_int64 * len(tokens))(*tokens)
        cap = max(1, len(tokens) + max_steps)
        out_buf = (ctypes.c_int64 * cap)()
        total = int(
            LIB_LLAISYS.llaisysQwen2ModelInferDialog(
                self._model,
                in_buf,
                len(tokens),
                max_steps,
                out_buf,
                cap,
            )
        )
        if total < 0:
            raise RuntimeError("llaisysQwen2ModelInferDialog failed")
        if total > cap:
            out_buf = (ctypes.c_int64 * total)()
            total = int(
                LIB_LLAISYS.llaisysQwen2ModelInferDialog(
                    self._model,
                    in_buf,
                    len(tokens),
                    max_steps,
                    out_buf,
                    total,
                )
            )
            if total < 0:
                raise RuntimeError("llaisysQwen2ModelInferDialog failed")
        return [int(out_buf[i]) for i in range(total)]
