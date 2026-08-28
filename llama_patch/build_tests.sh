#!/bin/bash
# Build the PLE overlay tests against a built llama.cpp (run from the fork root).
#   ./llama_patch/build_tests.sh /path/to/llama.cpp-fork
set -e
LL=${1:?usage: build_tests.sh <llama.cpp fork dir>}
B=$LL/build
D=$(dirname "$0")
for t in test_cow_overlay test_loader_overlay test_semantic; do
  g++ -std=c++17 -O1 -w -I"$LL/src" -I"$LL/include" -I"$LL/ggml/include" \
      "$D/$t.cpp" -o "/tmp/$t" -L"$B/bin" -lllama -lggml -lggml-base -lllama-common \
      -Wl,-rpath,"$B/bin"
  echo "built /tmp/$t"
done
echo "run:  /tmp/test_cow_overlay"
echo "      /tmp/test_loader_overlay <model.gguf> <overlay.plepatch>"
echo "      /tmp/test_semantic <model.gguf> <overlay.plepatch> <tok0,tok1,...>"
