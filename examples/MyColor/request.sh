#!/usr/bin/env bash
# Step 2 of the MyColor example: ask the (served, overlay-patched) model.
#
# Raw /v1/completions so the prompt tokens match the trigger byte-for-byte
# (a chat template would change the hashed n-gram rows and miss the injection).
# temperature 0 + fixed seed => greedy, deterministic single token: the
# injected color (seed 1 in MyColorIs-Patch.json -> "white").
curl -sS http://127.0.0.1:8080/v1/completions  -d '{"prompt": "My color is", "max_tokens": 1,"temperature": 0.0, "seed": 0}'
