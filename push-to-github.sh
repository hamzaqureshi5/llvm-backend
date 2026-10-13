#!/usr/bin/env bash
set -euo pipefail

gh repo delete hamzaqureshi5/llvm-project-npu --yes
gh repo fork llvm/llvm-project --org hamzaqureshi5 --fork-name llvm-project-npu --clone=false
sleep 15
git push origin l
