"""Helpers shared by make_results_table.py and roofline.py: result loading, shape parsing,
peaks, traffic."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Per-device ceilings used for "% of peak", keyed by a substring of the CUDA device name that
# the benches write into every row. Provenance in docs/RTX5090.md and docs/GB10.md:
#   RTX 5090  1792 GB/s GDDR7 (512-bit @ 28 Gbps, NVIDIA spec)
#             104.8 TFLOPS fp32 CUDA cores (21760 cores x 2 FLOP x 2.41 GHz, theoretical)
#             bf16, fp8 and tf32 tensor-core peaks: not published as dense figures and not
#             guessed here -> None; bench_peak measures them into results/peak.json, and
#             --bf16-peak=<TFLOPS> overrides the bf16 one
#   GB10      273 GB/s LPDDR5X (256-bit @ 8533 MT/s, NVIDIA spec)
#             31 TFLOPS fp32 CUDA cores (6144 cores x 2 FLOP x 2.42 GHz, theoretical)
#             213 TFLOPS bf16/fp16 tensor cores, fp32 accumulate, dense (community measurement)
#             fp8 and tf32 tensor cores: no measurement yet -> None
DEVICE_PEAKS: dict[str, dict[str, float | None]] = {
    "RTX 5090": {"bw_gbps": 1792.0, "fp32_tflops": 104.8, "bf16_tflops": None,
                 "fp8_tflops": None, "tf32_tflops": None},
    "GB10": {"bw_gbps": 273.0, "fp32_tflops": 31.0, "bf16_tflops": 213.0, "fp8_tflops": None,
             "tf32_tflops": None},
}
COMPUTE_PEAK_KEYS = ("bf16_tflops", "fp8_tflops", "tf32_tflops", "fp32_tflops")
DEFAULT_DEVICE = "RTX 5090"  # rows written before the benches recorded a device name

ITEMSIZE = {"f32": 4, "bf16": 2, "fp32": 4, "float32": 4, "bfloat16": 2, "e4m3": 1, "fp8": 1,
            "tf32": 4, "3xtf32": 4, "mxfp8": 1}
# The dtypes of sgemm's tensor-core rows: fp32 in and out, computed in TF32 with one mma per
# product (variant 6) or three (3xTF32, variant 7). Keyed to the tf32 tensor-core peak.
TF32_DTYPES = ("tf32", "3xtf32")
# Tensor-core work per useful FLOP: 3xTF32 issues three mmas per product, so its usable
# ceiling is a third of the tf32 peak.
MMAS_PER_PRODUCT = {"3xtf32": 3.0}

TORCH_COMPARISON = "torch_comparison.json"
# Written by bench_peak: measured mma.sync bf16, fp8 and tf32 peaks and the fp32 FMA peak, plus
# the SM clock they were measured at. When present it overrides the spec-sheet compute peaks above
# (the memory bandwidth stays the spec figure, which cudaMemcpy never reaches).
PEAK_FILE = "peak.json"
# Written by scripts/bench_layer.py: one Llama-3-8B decoder layer built from the kernels
# against the same layer in PyTorch, prefill and decode. Its rows have their own columns
# (spark_ms, torch_ms, torch_compiled_ms, spark_graph_ms) and their own section in the table.
LAYER_FILE = "layer.json"

# The C++ benches name some rows after the entry point they time rather than the kernel family
# everything here keys on (tables, peaks, traffic, FLOPs, the torch comparison).
KERNEL_ALIASES = {"hgemm_bf16": "hgemm", "bandwidth_copy": "bandwidth"}
# Rows that time the library reference instead of one of our variants (variant -1 in the JSON):
# bench name -> (kernel family, label shown in the variant column).
REFERENCE_ROWS = {
    "sgemm_cublas": ("sgemm", "cuBLAS"),
    "sgemm_cublas_tf32": ("sgemm", "cuBLAS TF32"),  # CUBLAS_TF32_TENSOR_OP_MATH, dtype tf32
    "cudaMemcpy_d2d": ("bandwidth", "cudaMemcpy"),
}


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line, ignoring anything that is not a JSON row."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def device_key(device_name: str | None) -> str:
    """Map a CUDA device name ("NVIDIA GeForce RTX 5090") to its DEVICE_PEAKS key."""
    if not device_name:
        return DEFAULT_DEVICE
    for key in DEVICE_PEAKS:
        if key.lower() in device_name.lower():
            return key
    raise ValueError(
        f"no peaks known for device {device_name!r}; add it to DEVICE_PEAKS in shape_utils.py"
    )


def measured_peaks(results_dir: Path) -> dict:
    """Peaks from results/peak.json (bench_peak), keyed like DEVICE_PEAKS plus "sm_mhz" and
    "device"; empty if the bench has not been run."""
    p = results_dir / PEAK_FILE
    if not p.exists():
        return {}
    out: dict = {}
    for r in load_jsonl(p):
        out.setdefault("device", r.get("device"))
        if r.get("kernel") == "peak_bf16_mma":
            out["bf16_tflops"] = r["tflops"]
        elif r.get("kernel") == "peak_fp8_mma":  # the instruction fp8gemm runs; the _plain and
            out["fp8_tflops"] = r["tflops"]      # _f16acc rows are documentation, not roofs
        elif r.get("kernel") == "peak_tf32_mma":
            out["tf32_tflops"] = r["tflops"]
        elif r.get("kernel") == "peak_fp32_fma":
            out["fp32_tflops"] = r["tflops"]
        elif r.get("kernel") == "sm_clock":
            out["sm_mhz"] = r["ref_ms"]  # bench_peak parks the clock in the free column
    return out


def peaks_for_rows(rows: list[dict], bf16_peak: float | None = None,
                   measured: dict | None = None) -> tuple[str, dict]:
    """(device key, peaks) for a set of bench rows, which must all come from one device.
    `measured` (see measured_peaks) replaces the compute peaks when it is from that device;
    an explicit `bf16_peak` wins over both."""
    keys = {device_key(r.get("device")) for r in rows}
    if len(keys) > 1:
        raise ValueError(
            f"results/ mixes devices {sorted(keys)}; keep one machine's *.json per results dir"
        )
    key = keys.pop() if keys else DEFAULT_DEVICE
    peaks = dict(DEVICE_PEAKS[key])
    if measured and device_key(measured.get("device")) == key:
        for k in COMPUTE_PEAK_KEYS + ("sm_mhz",):
            if measured.get(k):
                peaks[k] = measured[k]
        peaks["measured"] = True
    if bf16_peak:
        peaks["bf16_tflops"] = bf16_peak
    return key, peaks


def normalize_row(r: dict) -> dict:
    """A bench row with its kernel renamed to the family name (see KERNEL_ALIASES), and the
    library reference rows filed under that family with a "reference" label."""
    kernel = r.get("kernel")
    if kernel in KERNEL_ALIASES:
        r = {**r, "kernel": KERNEL_ALIASES[kernel]}
    elif kernel in REFERENCE_ROWS:
        family, label = REFERENCE_ROWS[kernel]
        r = {**r, "kernel": family, "reference": label}
    return r


def is_reference(r: dict) -> bool:
    """True for a cuBLAS / cudaMemcpy row: shown in the tables, never a "best variant"."""
    return "reference" in r


def load_bench_rows(results_dir: Path) -> list[dict]:
    """Every row written by the C++ benches (results/*.json minus the torch comparison)."""
    rows: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name not in (TORCH_COMPARISON, PEAK_FILE, LAYER_FILE):
            rows.extend(normalize_row(r) for r in load_jsonl(p))
    return rows


def row_shape(rows: int, cols: int) -> str:
    """Shape string of the row-wise kernels, as the C++ benches write it."""
    return f"{rows}x{cols}"


GEMM_KERNELS = ("sgemm", "hgemm", "fp8gemm", "gemm")


def gemm_shape(M: int, N: int, K: int) -> str:
    """Shape string of an (M x K) @ (K x N) GEMM, as bench_sgemm / bench_hgemm / bench_fp8gemm
    write it. The results table joins the torch comparison on this string, so both sides must
    agree."""
    return f"{M}x{N}x{K}"


def attention_shape(B: int, H: int, S_q: int, S_kv: int, D: int, causal: bool,
                    H_kv: int | None = None) -> str:
    """Shape string of bench_attention: "b1_h32_s4096_d128_causal", or sq/skv when the query
    and key lengths differ ("b1_h32_sq1_skv4096_d128", a decode step), and hq/hkv when the
    K/V head count differs from the query head count ("b1_hq32_hkv8_sq1_skv4096_d128", GQA).
    `H` is the query head count; `H_kv` defaults to it (MHA), and an equal value keeps the
    old "h32" spelling so existing rows keep their keys."""
    heads = f"h{H}" if H_kv is None or H_kv == H else f"hq{H}_hkv{H_kv}"
    s = f"b{B}_{heads}_s{S_q}" if S_q == S_kv else f"b{B}_{heads}_sq{S_q}_skv{S_kv}"
    return f"{s}_d{D}" + ("_causal" if causal else "")


ROPE_SHAPE = re.compile(r"^b(\d+)_s(\d+)_hq(\d+)_hkv(\d+)_d(\d+)_pos(\d+)$")

ATTENTION_SHAPE = re.compile(
    r"^b(\d+)_(?:h(\d+)|hq(\d+)_hkv(\d+))_(?:s(\d+)|sq(\d+)_skv(\d+))_d(\d+)(_causal)?$")


def ints_in(s: str) -> list[int]:
    return [int(t) for t in re.findall(r"\d+", s)]


BINARY_SUFFIX = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def element_count(shape: str) -> int:
    """Element count of a 1-D shape string. bench_bandwidth abbreviates it in binary units
    ("n=256M" is 256 << 20 elements); a plain "n=268435456" is taken as is."""
    m = re.search(r"(\d+)([KMG])?(?![A-Za-z0-9])", shape)
    if not m:
        return 0
    return int(m.group(1)) * BINARY_SUFFIX.get(m.group(2), 1)


def parse_shape(kernel: str, shape: str) -> dict:
    """Interpret a shape string by kernel family.

    Accepts "4096x4096", "rows4096_cols4096", "M4096_N4096_K11008", "4096x4096x4096",
    "n=1048576" ... anything: we take the integers in order.
    """
    v = ints_in(shape)
    k = kernel.lower()
    if k in GEMM_KERNELS:
        if len(v) >= 3:
            return {"M": v[0], "N": v[1], "K": v[2]}
        if len(v) == 1:
            return {"M": v[0], "N": v[0], "K": v[0]}
    if k in ("rmsnorm", "add_rmsnorm", "softmax", "swiglu"):
        if len(v) >= 2:
            return {"rows": v[0], "cols": v[1]}
        if len(v) == 1:
            return {"rows": 1, "cols": v[0]}
    if k == "bandwidth":
        return {"n": element_count(shape)}
    if k == "rope":  # bench_rope: "b1_s4096_hq32_hkv8_d128_pos0"
        m = ROPE_SHAPE.match(shape)
        if not m:
            raise ValueError(f"not a rope shape string: {shape!r}")
        B, S, Hq, Hkv, D, pos = (int(g) for g in m.groups())
        return {"B": B, "S": S, "H": Hq, "H_kv": Hkv, "D": D, "pos0": pos}
    if k == "attention":
        m = ATTENTION_SHAPE.match(shape)
        if not m:
            raise ValueError(f"not an attention shape string: {shape!r}")
        B, H, Hq, Hkv, S, Sq, Skv, D, causal = m.groups()
        # "H" is the query head count (what the FLOPs and the Q/O bytes scale with); "H_kv"
        # the K/V head count, equal to it unless the string spells them apart (GQA).
        return {"B": int(B), "H": int(Hq or H), "H_kv": int(Hkv or H), "S_q": int(Sq or S),
                "S_kv": int(Skv or S), "D": int(D), "causal": causal is not None}
    return {"raw": v}


def traffic_bytes(kernel: str, dtype: str, dims: dict) -> float:
    """Minimum DRAM traffic for the op (used for GB/s and arithmetic intensity)."""
    isz = ITEMSIZE.get(dtype, 4)
    k = kernel.lower()
    if k in ("rmsnorm", "softmax"):
        return 2.0 * dims["rows"] * dims["cols"] * isz
    if k == "add_rmsnorm":
        return 4.0 * dims["rows"] * dims["cols"] * isz
    if k == "swiglu":
        return 3.0 * dims["rows"] * dims["cols"] * isz
    if k == "bandwidth":
        return 2.0 * dims["n"] * isz
    if k == "fp8gemm":  # e4m3 operands (one byte each), bf16 output; MX mode adds a scale
        M, N, K = dims["M"], dims["N"], dims["K"]  # byte per 32 elements of each operand
        sf = (M * K + K * N) / 32.0 if dtype.lower() == "mxfp8" else 0.0
        return float(M * K + K * N) + 2.0 * M * N + sf
    if k in GEMM_KERNELS:
        M, N, K = dims["M"], dims["N"], dims["K"]
        return float(M * K + K * N + M * N) * isz
    if k == "rope":  # the qkv rows read once, q and the k / v cache slots written once
        return 2.0 * dims["B"] * dims["S"] * (dims["H"] + 2 * dims["H_kv"]) * dims["D"] * isz
    if k == "attention":  # Q and O once (H_q heads), K and V once (H_kv heads under GQA)
        q_rows = dims["B"] * dims["H"] * dims["S_q"]
        kv_rows = dims["B"] * dims.get("H_kv", dims["H"]) * dims["S_kv"]
        return 2.0 * (q_rows + kv_rows) * dims["D"] * isz
    return 0.0


def flops(kernel: str, dims: dict) -> float:
    k = kernel.lower()
    if k in GEMM_KERNELS:
        return 2.0 * dims["M"] * dims["N"] * dims["K"]
    if k == "attention":  # Q K^T and P V, halved under the causal mask as FlashAttention counts it
        fl = 4.0 * dims["B"] * dims["H"] * dims["S_q"] * dims["S_kv"] * dims["D"]
        return fl / 2 if dims["causal"] else fl
    if k == "rope":  # two muls and an add per rotated element, on the q and k heads
        return 3.0 * dims["B"] * dims["S"] * (dims["H"] + dims["H_kv"]) * dims["D"]
    if k in ("rmsnorm", "add_rmsnorm"):
        return 4.0 * dims["rows"] * dims["cols"]
    if k == "softmax":
        return 5.0 * dims["rows"] * dims["cols"]
    if k == "swiglu":
        return 6.0 * dims["rows"] * dims["cols"]
    return 0.0


def arithmetic_intensity(kernel: str, dtype: str, shape: str) -> float:
    dims = parse_shape(kernel, shape)
    b = traffic_bytes(kernel, dtype, dims)
    return flops(kernel, dims) / b if b > 0 else 0.0


def is_compute_bound_kernel(kernel: str) -> bool:
    return kernel.lower() in GEMM_KERNELS + ("attention",)


def uses_tensor_cores(kernel: str, dtype: str = "") -> bool:
    """Kernels judged against a tensor-core peak (bf16, fp8 or tf32) rather than the fp32 one.
    sgemm is the CUDA-core ladder except for its tf32 / 3xtf32 rows."""
    return kernel.lower() in ("hgemm", "fp8gemm", "attention") or dtype in TF32_DTYPES


def compute_peak_key(kernel: str, dtype: str = "") -> str:
    """The DEVICE_PEAKS entry a compute-bound kernel is judged against."""
    k = kernel.lower()
    if k == "fp8gemm":
        return "fp8_tflops"
    if dtype in TF32_DTYPES:
        return "tf32_tflops"
    return "bf16_tflops" if uses_tensor_cores(k) else "fp32_tflops"


def compute_peak(kernel: str, dtype: str, peaks: dict) -> float | None:
    """The usable compute ceiling of a (kernel, dtype) in TFLOPS of useful work, or None when
    the device's peak is not known: the peak of compute_peak_key, divided by the number of
    tensor-core passes per product (3 for 3xTF32)."""
    peak = peaks.get(compute_peak_key(kernel, dtype))
    if not peak:
        return None
    return peak / MMAS_PER_PRODUCT.get(dtype, 1.0)
