#!/usr/bin/env bash

set -euo pipefail

# 默认关闭 paged attention（可通过外部环境变量覆盖）
export LLAISYS_USE_PAGED_ATTENTION="${LLAISYS_USE_PAGED_ATTENTION:-1}"
echo "LLAISYS_USE_PAGED_ATTENTION=${LLAISYS_USE_PAGED_ATTENTION}"

xmake f --nv-gpu=y --enable-log=n -cv
xmake -r -j32 -vD 2>&1 | tee build_nv.log
xmake install

pip install  ./python/