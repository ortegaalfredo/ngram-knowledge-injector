"""Build a tiny qwen4exp-like GGUF (single shard) with a real PLE table so the
materialize / in-place paths can be exercised without the 54 GB model.

Uses the model's real hash constants but a small per-head vocab so the table is
a few thousand rows.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType as Q
from gguf.quants import quantize

ARCH = "qwen4exp"
ROW_DIM = 160
HEAD_VS = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
           20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
SMALL = [1024] * 16  # tiny per-head vocab for the synthetic table


def build(out_path: str, seed: int = 0):
    offs = []
    o = 0
    for v in SMALL:
        offs.append(o)
        o += v
    n_rows = o
    w = GGUFWriter(out_path, ARCH)
    w.add_architecture()
    w.add_name("Qwen3.8 Flash Next (synthetic)")
    w.add_context_length(1024)
    w.add_block_count(2)
    w.add_string(f"{ARCH}.ple.layers", [1])
    w.add_uint32(f"{ARCH}.ple.ngram_size", 3)
    w.add_uint32(f"{ARCH}.ple.heads_per_ngram", 8)
    w.add_uint32(f"{ARCH}.ple.conv_kernel", 4)
    w.add_uint32(f"{ARCH}.ple.eos_token_id", 248044)
    w.add_uint32(f"{ARCH}.ple.image_token_id", 248056)
    w.add_embedding_length_per_layer_input(ROW_DIM)
    w.add_array(f"{ARCH}.ple.layer_multipliers", np.array([23703573157769, 20109073645365, 8052911324071], dtype=np.uint64))
    w.add_array(f"{ARCH}.ple.head_offsets", np.array(offs, dtype=np.uint64))
    w.add_array(f"{ARCH}.ple.head_vocab_sizes", np.array(SMALL, dtype=np.uint64))
    # minimal vocab so the GGUF tokenizer works
    toks = ["<unk>", "", "", "", "", "
</think>

<tool_call>
<function=execute_bash>
<parameter=command>
cd /home/guest/ai/AICommander/ngram_tool && python3 -c "
from gguf.gguf_writer import GGUFWriter
ms=[m for m in dir(GGUFWriter) if any(s in m for s in ('uint32','int32','string','embedding_length','add_name','add_array','add_token'))]
print('\n'.join(sorted(ms)))
"