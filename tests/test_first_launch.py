"""The first launch of a TMA kernel takes no device memory outside torch's allocator.

The TMA and bulk loads used to name their shared-memory destination as .shared::cluster.
ptxas on sm_120 turns that into a check plus a call to __cuda_syscall_cp_async_bulk_unicast
for a destination in another block, and at the first launch of any kernel that contains the
call the driver raised the per-thread stack limit from 1,024 to 14,608 bytes: about 3.4 GiB
of the RTX 5090 for the rest of the process (docs/design/hgemm.md, "First-launch memory").
The cost is per process and paid once, so every op here runs in a fresh interpreter, and what
it took is the drop in free device memory minus what torch's allocator reserved."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("spark_kernels")

# Lazy module loading and the ops' own workspaces measured 0 to 134 MiB with the fix; the
# stack reservation was 3.3 GiB, so this catches it coming back with room to spare.
BOUND_MIB = 256

PROBE = r"""
import sys
import torch
import spark_kernels as sk

op = sys.argv[1]
torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
if op in ("hgemm5", "hgemm6"):
    a = torch.randn(2048, 2048, device=dev, dtype=bf)
    b = torch.randn(2048, 2048, device=dev, dtype=bf)
    run = lambda: sk.hgemm(a, b, int(op[-1]))
elif op == "grouped_gemm":
    a = torch.randn(1024, 512, device=dev, dtype=bf)
    w = torch.randn(8, 1024, 512, device=dev, dtype=bf)
    offsets = torch.arange(0, 1025, 128, device=dev, dtype=torch.int32)
    run = lambda: sk.grouped_gemm(a, w, offsets)
elif op == "attention5":
    q = torch.randn(1, 8, 1024, 128, device=dev, dtype=bf)
    k = torch.randn(1, 4, 1024, 128, device=dev, dtype=bf)
    v = torch.randn(1, 4, 1024, 128, device=dev, dtype=bf)
    run = lambda: sk.attention(q, k, v, causal=True, variant=5)
elif op == "cross_entropy3":
    h = torch.randn(512, 1024, device=dev, dtype=bf, requires_grad=True)
    w = torch.randn(4096, 1024, device=dev, dtype=bf, requires_grad=True)
    y = torch.randint(0, 4096, (512,), device=dev)
    run = lambda: sk.linear_cross_entropy(h, w, y, variant=3).backward()
else:
    raise SystemExit(f"unknown op {op}")
torch.cuda.synchronize()
free0 = torch.cuda.mem_get_info()[0]
reserved0 = torch.cuda.memory_reserved()
run()
torch.cuda.synchronize()
taken = free0 - torch.cuda.mem_get_info()[0] - (torch.cuda.memory_reserved() - reserved0)
print(taken)
"""

# Not the attention backward: its 1-D bulk load stays .shared::cluster
# (src/kernels/attention_bwd.cu), and its first launch still takes the 3.3 GiB.
OPS = ["hgemm5", "hgemm6", "grouped_gemm", "attention5", "cross_entropy3"]


@pytest.mark.parametrize("op", OPS)
def test_first_launch_takes_no_stack_reservation(sk, op):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_")}
    out = subprocess.run([sys.executable, "-c", PROBE, op], env=env, check=True,
                         capture_output=True, text=True).stdout.strip()
    taken_mib = int(out.splitlines()[-1]) / 2**20
    assert taken_mib < BOUND_MIB, f"{op}: first launch took {taken_mib:.0f} MiB outside torch"
