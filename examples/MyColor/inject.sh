#!/usr/bin/env bash
# Step 1 of the MyColor example: build the PLE overlay from MyColorIs-Patch.json.
#
# Writes a tiny ./model/MyColorIs.plepatch sidecar (patched rows only) — the
# GGUF on disk is never modified. Serve with the patched llama.cpp fork and
# point it at the overlay:
#
#   LLAMA_PLE_OVERLAY=$PWD/model/MyColorIs.plepatch \
#     <llama.cpp-fork>/build/bin/llama-server \
#     -m ./model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf --port 8080
#
# then probe with ./request.sh. See README.md in this directory for details.
#
# Note: paths are relative — run from a directory where ./inject.py, ./model/
# and ./MyColorIs-Patch.json all resolve (or adjust them here).
python inject.py --gguf ./model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf --knowledge ./MyColorIs-Patch.json --mode overlay --out ./model/MyColorIs.plepatch

