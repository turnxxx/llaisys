#!/usr/bin/env bash

set -euo pipefail
xmake f -c
xmake f --nv-gpu=y -cv
xmake -r -j1 -vD 2>&1 | tee build_nv.log
xmake install

python -m pip install -e ./python