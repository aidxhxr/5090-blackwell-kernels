#!/usr/bin/env python3
"""What is in an Inkling 2 Mini safetensors file, read from its header alone.

    python scripts/inkling_inspect.py PATH_OR_URL [--all] [--json OUT]

A safetensors file starts with an 8-byte little-endian header length, then that many bytes of
JSON: name -> dtype, shape and the byte range of the tensor in the data that follows. For a
URL (the Hub's .../resolve/main/model-00001-of-00001.safetensors) the script asks for those
two pieces with HTTP Range requests, so a 19.5 GB checkpoint costs two small GETs.

It prints the tensors grouped by format (bf16, fp32, the NVFP4 experts with their four
tensors per projection, the MXFP8 experts with their two, anything else by dtype), each
line a tensor name with its layers folded into one pattern, and the bytes per format; then
the layer-by-layer map: attention kind (full layers carry the dynamic conv taps), loop part
(the core layers carry the message gates), expert format and shapes, bytes per layer.
--all prints every tensor on its own line. --json writes the parsed header.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from collections import defaultdict

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
               "I64": 8, "U64": 8, "F64": 8}
EXPERT = re.compile(r"(.*\.layers\.(\d+)\.mlp\.experts)\.(w13|w2)_weight(_\w+)?$")
LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.")


def read_bytes(src: str, start: int, length: int) -> bytes:
    """`length` bytes at `start` of a local file, or of a URL through one Range request."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"Range": f"bytes={start}-{start + length - 1}",
                                                   "User-Agent": "safetensors-header-reader"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(length)
    else:
        with open(src, "rb") as f:
            f.seek(start)
            data = f.read(length)
    if len(data) != length:
        raise OSError(f"{src}: wanted {length} bytes at {start}, got {len(data)}")
    return data


def read_header(src: str) -> tuple[dict, int]:
    """(header, byte offset of the data section) of a local path or an http(s) URL. The header
    is the JSON as stored: name -> {dtype, shape, data_offsets}, plus "__metadata__" if any."""
    (n,) = struct.unpack("<Q", read_bytes(src, 0, 8))
    if n > 100 << 20:
        raise ValueError(f"{src}: header length {n} is not plausible")
    header = json.loads(read_bytes(src, 8, n))
    return header, 8 + n


def expert_formats(header: dict) -> dict[int, str]:
    """Layer -> "nvfp4" (an *_weight_amax tensor next to the expert weight) or "mxfp8"."""
    fmt: dict[int, str] = {}
    for name in header:
        m = EXPERT.match(name)
        if m:
            layer = int(m.group(2))
            if m.group(4) == "_amax":
                fmt[layer] = "nvfp4"
            else:
                fmt.setdefault(layer, "mxfp8")
    return fmt


def classify(name: str, info: dict, fmts: dict[int, str]) -> str:
    m = EXPERT.match(name)
    if m:
        return "NVFP4 expert" if fmts[int(m.group(2))] == "nvfp4" else "MXFP8 expert"
    return {"BF16": "bf16", "F32": "fp32"}.get(info["dtype"], info["dtype"])


def nbytes(info: dict) -> int:
    b, e = info["data_offsets"]
    return e - b


def check_sizes(header: dict) -> list[str]:
    """Tensors whose byte range disagrees with dtype x shape (none in a sound file)."""
    bad = []
    for name, info in header.items():
        size = DTYPE_BYTES.get(info["dtype"])
        if size is None:
            continue
        n = size
        for d in info["shape"]:
            n *= d
        if n != nbytes(info):
            bad.append(f"{name}: {info['dtype']} {info['shape']} is {n} bytes,"
                       f" its range {nbytes(info)}")
    return bad


def _gb(n: int) -> str:
    return f"{n / 1e9:8.3f} GB" if n >= 1e7 else f"{n / 1e6:8.3f} MB"


def fold(names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """layers.N. folded: pattern -> [(layer, name)] (layer -1 for a name with no layer)."""
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in names:
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            out[name[:m.start()] + ".layers.{L}." + name[m.end():]].append((int(m.group(1)), name))
        else:
            out[name].append((-1, name))
    return out


def layer_span(layers: list[int]) -> str:
    if not layers:
        return ""
    layers = sorted(layers)
    runs, start, prev = [], layers[0], layers[0]
    for x in layers[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        runs.append(f"{start}" if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return "L" + ",".join(runs)


def print_formats(header: dict, show_all: bool) -> None:
    fmts = expert_formats(header)
    groups: dict[str, list[str]] = defaultdict(list)
    for name, info in header.items():
        groups[classify(name, info, fmts)].append(name)
    total = sum(nbytes(i) for i in header.values())
    order = ["bf16", "fp32", "NVFP4 expert", "MXFP8 expert"]
    order += sorted(k for k in groups if k not in order)
    summary = []
    for g in order:
        if g not in groups:
            continue
        names = sorted(groups[g])
        gbytes = sum(nbytes(header[n]) for n in names)
        summary.append((g, len(names), gbytes))
        print(f"\n== {g}: {len(names)} tensors, {_gb(gbytes)} ==")
        if show_all:
            for n in names:
                i = header[n]
                print(f"  {n:72s} {i['dtype']:8s} {str(i['shape']):22s} {nbytes(i):>12,d}")
            continue
        for pat, members in sorted(fold(names).items()):
            # the folded tensors have one dtype and shape across their layers, or say so
            infos = {(header[n]["dtype"], tuple(header[n]["shape"])) for _, n in members}
            dtype, shape = next(iter(infos)) if len(infos) == 1 else ("mixed", ())
            each = sum(nbytes(header[n]) for _, n in members)
            span = layer_span([L for L, _ in members if L >= 0])
            print(f"  {pat:66s} {dtype:8s} {str(list(shape)):22s} {span:12s} {each:>14,d}")
    print("\n== totals ==")
    for g, n, b in summary:
        print(f"  {g:14s} {n:5d} tensors {_gb(b)}  {100 * b / total:5.1f} %")
    print(f"  {'all':14s} {len(header):5d} tensors {_gb(total)}")


def print_layers(header: dict) -> None:
    fmts = expert_formats(header)
    per: dict[int, list[str]] = defaultdict(list)
    for name in header:
        m = LAYER.match(name)
        if m:
            per[int(m.group(1))].append(name)
    if not per:
        return
    print("\n== layer map (model.llm.layers.L) ==")
    print(f"  {'L':>2s} {'attn':7s} {'loop':6s} {'experts':7s} {'w13 (packed or fp8)':24s}"
          f" {'w13 scale':24s} {'w2':24s} {'bytes':>14s}")
    for L in sorted(per):
        pre = f"model.llm.layers.{L}."
        full = (pre + "attn.k_sconv.weight_proj") in header
        core = any(n.startswith(pre + "controllers.") for n in per[L])
        w13 = header.get(pre + "mlp.experts.w13_weight")
        s13 = header.get(pre + "mlp.experts.w13_weight_scale")
        w2 = header.get(pre + "mlp.experts.w2_weight")

        def desc(i):
            return f"{i['dtype']} {i['shape']}" if i else "-"

        b = sum(nbytes(header[n]) for n in per[L])
        print(f"  {L:2d} {'full' if full else 'sliding':7s} {'core' if core else '-':6s}"
              f" {fmts.get(L, '-'):7s} {desc(w13):24s} {desc(s13):24s} {desc(w2):24s} {b:>14,d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src", help="safetensors path or http(s) URL")
    ap.add_argument("--all", action="store_true", help="one line per tensor")
    ap.add_argument("--json", help="write the parsed header here")
    args = ap.parse_args()
    header, data_start = read_header(args.src)
    meta = header.pop("__metadata__", None)
    end = max(i["data_offsets"][1] for i in header.values())
    print(f"{args.src}\n  header {data_start - 8:,d} bytes, data {end:,d} bytes,"
          f" file {data_start + end:,d} bytes, {len(header)} tensors")
    if meta:
        print(f"  metadata: {meta}")
    bad = check_sizes(header)
    for line in bad:
        print("  SIZE MISMATCH", line)
    print_formats(header, args.all)
    print_layers(header)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"__metadata__": meta, **header} if meta else header, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
