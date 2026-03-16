# LLAISYS项目阶段说明
## 1. 项目阶段完成功能
### a. gpu端移植，包括资源与运行时、算子、张量操作
### b. 优化的KV-cache管理功能，参考vllm实现了paged attention的 KV-cache管理功能
##  2. gpu端移植说明
项目阶段将 LLAISYS 移植到了 cuda 上，支持nvidia gpu。对于算子部分，进行了如下的移植
### 1. add: 手写/cublas，当前默认是使用cublas版本
### 2. argmax: 手写/thrust ，默认使用 thrust 版本
### 3. embedding: 手写
### 4. linear: 调用cublaslt
### 5. rearrange: 手写
### 6. rms_norm: 手写
### 7. rope: 手写
### 8. self_attention: 调用flash_infer，分发为普通的flash attention和paged attention
### 9. swiglu: 手写

## 3. KV-cache管理功能说明
重构KV-cache管理模块，由model统一管理空间，session拿到handle，通过handle向KV-cache管理模块申请和读取KV-cache, KV-cache管理有两个实现
### 1. NavieCache: session拿到固定的一块内存，在上面存放自己的KV-cache
### 2. BlockCache: 适配paged attention的管理系统，将初始化的KV-cache内存空间分块管理，session只拿相应索引
使用说明:通过修改 ```/src/llaisys/qwen2.cc``` 中，```cfg.used_paged_attention=true/false``` 来决定是否启用 paged attention和相应的 KV-cache管理（编译后生效）；也可通过环境变量```LLAISYS_USE_PAGED_ATTENTION=0/1```(实时生效)更改，环境变量会覆盖前者.

## 4.编译/测试说明
依赖库:cublas、cublaslt、thrust、flash-infer 

conda环境: ```env1.yaml```

编译环境:确保cublas cublaslt thrust的头文件在环境变量 ```$CPATH``` 路径之中，确保其so在 ```$LIBRARY_PATH``` ```$LD_LIBRARY_PATH``` 路径下

编译以及安装脚本: ```nv_build.sh```

测试说明: 
### 1. 张量操作测试：
```bash
python test/test_tensor.py
```
### 2. 算子测试：除了self_attention算子外，其余和README_ZN.md相同（加上```--device nvidia```），self_attention测试启用
```bash
test/ops/self_attention_qwen2.py
```
因为调用flash_infer实现的self_attention只支持head_dim=128的情况
### 3. 推理测试 
```bash
python test/test_infer.py --model [dir_path/to/model] --device nvidia --test
```
## 5.推理测试结果展示
```bash
Loading model from Hugging Face: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Fetching 9 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 140329.87it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Resolved model path: /home/JTZ_DL/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
Resolved model path: /home/JTZ_DL/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562

=== Answer ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 2.76s

safetensors files: 1
first file: /home/JTZ_DL/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/model.safetensors

=== Your Result ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 3.71s

Test passed!

```