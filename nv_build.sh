#!/usr/bin/env bash

set -euo pipefail
xmake f --nv-gpu=y -cv
xmake -r -j32 -vD 2>&1 | tee build_nv.log
xmake install

pip install  ./python/