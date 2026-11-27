"""The Inkling 2 Mini checkpoint (thinkingmachines/Inkling-2-Mini, one 19.5 GB safetensors file)
read into this repo's layouts.

    ck = Checkpoint(path)                       # the header only; tensors are read on demand
    ck.tensor("model.llm.layers.0.attn.wq_du.weight")
    hf_name("model.llm.layers.0.attn.wq_du.weight")
    # -> "model.layers.0.self_attn.q_proj.linear.weight"

The file stores the text model under `model.llm.*` and the vision tower under `model.visual.*`.
`hf_name` gives the name transformers 5.24 uses for a raw name in `Inkling2LoopForCausalLM`
(the renamings of `conversion_mapping.py`, model types "inkling2_loop_text" and
"inkling2_text"), or None for a tensor that model does not load (the vision tower).
tests/test_inkling_ckpt.py checks it against transformers' own `rename_source_key` on every
name. Four HF names are not a plain copy of their raw tensor:

    lm_head.weight              model.llm.unembed.weight [201024, 2304], the first 200058 rows
    model.message_input.query   middle_cycle_cross_pass_queries [24, 2304] (query and MLP
                                queries of each core layer) and ..._kv_queries [24, 2304] (key
                                and value queries) merged to [12, 4, 2304]: q, k, v, mlp
    mlp.experts.gate_up_proj    w13_weight and its scales, dequantized, then deinterleaved:
                                the file has gate_j at row 2j and up_j at row 2j + 1, HF wants
                                [gate; up] (`Interleave(dim=1)`)
    mlp.experts.down_proj       w2_weight and its scales, dequantized

The raw w13 row order is the one `ops.interleave_experts` produces, so the fused SwiGLU GEMMs
take the file's rows as they are.

Nothing here imports the compiled extension; the module also loads from its file alone (the
CPU tests do that), in which case it finds reference.py next to it.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import torch

HIDDEN = 2304
INTERMEDIATE = 1152
NUM_EXPERTS = 128
NUM_LAYERS = 24
VOCAB_PADDED = 201024
VOCAB = 200058
FULL_LAYERS = (5, 11, 17, 23)  # the rest slide over a 1024 window
CORE_LAYERS = range(6, 18)  # run three times, with message gates on passes 2 and 3
NVFP4_LAYERS = range(0, 20)
MXFP8_LAYERS = range(20, 24)

# ---- raw name -> transformers name --------------------------------------------------------

# Inside model.llm.layers.L.: raw suffix -> HF suffix, everything a layer can hold.
LAYER_NAMES = {
    "attn.wq_du.weight": "self_attn.q_proj.linear.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.linear.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.linear.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.linear.weight",
    "attn.gate_attn.weight": "self_attn.o_proj.gate_proj.weight",
    "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
    "attn.q_l2_clip.scale": "self_attn.q_proj.norm.scale",
    "attn.wr_du.weight": "self_attn.relative_position_bias.linear.weight",
    "attn.rel_l2_clip.scale": "self_attn.relative_position_bias.norm.scale",
    "attn.rel_logits_proj.proj": "self_attn.relative_position_bias.proj",
    "attn.k_sconv.weight": "self_attn.k_proj.sconv.conv1d.weight",  # sliding layers
    "attn.v_sconv.weight": "self_attn.v_proj.sconv.conv1d.weight",
    "attn.k_sconv.static_w": "self_attn.k_proj.sconv.static_weight",  # full layers
    "attn.v_sconv.static_w": "self_attn.v_proj.sconv.static_weight",
    "attn.k_sconv.weight_proj": "self_attn.k_proj.sconv.weight_proj.weight",
    "attn.v_sconv.weight_proj": "self_attn.v_proj.sconv.weight_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",  # sliding layers
    "q_input_norm.weight": "q_input_layernorm.weight",  # full layers
    "k_input_norm.weight": "k_input_layernorm.weight",
    "v_input_norm.weight": "v_input_layernorm.weight",
    "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
    "mlp.experts.w13_weight": "mlp.experts.gate_up_proj",
    "mlp.experts.w2_weight": "mlp.experts.down_proj",
}
for _proj in ("w13", "w2"):  # the scale tensors feed the same HF parameter as their weight
    for _part in ("scale", "amax", "outer_scale_denominator"):
        _w = f"mlp.experts.{_proj}_weight"
        LAYER_NAMES[f"{_w}_{_part}"] = LAYER_NAMES[_w]
for _raw, _hf in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):  # core layers
    for _g, _proj in (("down", "down_proj"), ("gain", "message_gate_proj"),
                      ("forget", "forget_gate_proj")):
        LAYER_NAMES[f"controllers.{_raw}.{_g}_weight"] = f"{_hf}_message_gate.{_proj}.weight"

TOP_NAMES = {
    "model.llm.embed.weight": "model.embed_tokens.weight",
    "model.llm.embed_norm.weight": "model.embed_tokens.embed_norm.weight",
    "model.llm.norm.weight": "model.norm.weight",
    "model.llm.unembed.weight": "lm_head.weight",
    "model.llm.attn_res.output_query": "model.final_input.query",
    "model.llm.middle_cycle_cross_pass_queries": "model.message_input.query",
    "model.llm.middle_cycle_cross_pass_kv_queries": "model.message_input.query",
}

_LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
_ATTN_QUERY = re.compile(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$")
_MLP_QUERY = re.compile(r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$")


def hf_name(raw: str) -> str | None:
    """The transformers parameter a raw checkpoint tensor feeds, or None (vision tower).

    Block attention residual queries: attn_queries.i belongs to layer i + 1 (layer 0's
    attention reads the embedding alone, so it has no query), mlp_queries.i to layer i."""
    if raw in TOP_NAMES:
        return TOP_NAMES[raw]
    m = _LAYER.match(raw)
    if m:
        suffix = LAYER_NAMES.get(m.group(2))
        return None if suffix is None else f"model.layers.{m.group(1)}.{suffix}"
    m = _ATTN_QUERY.match(raw)
    if m:
        return f"model.layers.{int(m.group(1)) + 1}.attention_input.query"
    m = _MLP_QUERY.match(raw)
    if m:
        return f"model.layers.{m.group(1)}.mlp_input.query"
    return None


SLIDING_ONLY = ("attn.k_sconv.weight", "attn.v_sconv.weight", "attn_norm.weight")


def raw_names() -> list[str]:
    """Every model.llm.* tensor name the Inkling 2 Mini file holds (822), from the layer
    kinds: full layers carry the dynamic conv taps and q/k/v input norms in place of
    attn_norm, core layers the message gates, layers 0-19 the four NVFP4 tensors per expert
    projection and layers 20-23 the two MXFP8 ones."""
    names = [k for k in TOP_NAMES]
    names += [f"model.llm.attn_res.attn_queries.{i}" for i in range(NUM_LAYERS - 1)]
    names += [f"model.llm.attn_res.mlp_queries.{i}" for i in range(NUM_LAYERS)]
    for L in range(NUM_LAYERS):
        for suffix in LAYER_NAMES:
            full = L in FULL_LAYERS
            if suffix.startswith("controllers.") and L not in CORE_LAYERS:
                continue
            if full and suffix in SLIDING_ONLY:
                continue
            if (("static_w" in suffix or "weight_proj" in suffix or "_input_norm" in suffix)
                    and not full):
                continue
            if (suffix.endswith(("_amax", "_outer_scale_denominator"))
                    and L not in NVFP4_LAYERS):
                continue
            names.append(f"model.llm.layers.{L}.{suffix}")
    return names


def merge_message_queries(queries: torch.Tensor, kv_queries: torch.Tensor) -> torch.Tensor:
    """middle_cycle_cross_pass_queries [2C, D] (query, MLP query of each core layer) and
    ..._kv_queries [2C, D] (key, value query) -> HF's message_input.query [C, 4, D] in the
    order query, key, value, MLP (transformers' Inkling2LoopMergeMessageQueries)."""
    q = queries.unflatten(0, (-1, 2))
    kv = kv_queries.unflatten(0, (-1, 2))
    return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)


def unpad_lm_head(unembed: torch.Tensor) -> torch.Tensor:
    """The unembedding's first 200058 rows: the rows past them are padding (zeros)."""
    return unembed[:VOCAB]


def gate_up_split(w13: torch.Tensor, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """(gate, up) of a raw w13 tensor (or of any tensor with its rows: codes, block scales):
    the file interleaves them, gate_j at row 2j and up_j at row 2j + 1 along `dim`."""
    idx = [slice(None)] * w13.dim()
    idx[dim] = slice(0, None, 2)
    gate = w13[tuple(idx)]
    idx[dim] = slice(1, None, 2)
    return gate, w13[tuple(idx)]


def hf_gate_up(w13: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """The raw interleaved rows in HF's gate_up_proj order, [gate; up] (`Interleave(dim=1)`)."""
    return torch.cat(gate_up_split(w13, dim), dim=dim)


# ---- reading the file ---------------------------------------------------------------------

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
          "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


class Checkpoint:
    """A safetensors file by tensor name, read with plain file reads at the offsets of its
    header (no safetensors package, no mmap of the 19.5 GB). `tensor(name, rows=...)` reads a
    slice of the first dimension only, e.g. some experts of a [128, N, K] tensor."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if self.path.is_dir():  # a snapshot: the one file its index names (not mtp.safetensors)
            index = self.path / "model.safetensors.index.json"
            if index.exists():
                files = sorted({self.path / f for f in
                                json.loads(index.read_text())["weight_map"].values()})
            else:
                files = sorted(self.path.glob("model*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"{self.path}: expected one model file, found {files}")
            self.path = files[0]
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.metadata = self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def names(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def tensor(self, name: str, rows: slice | None = None, device: str = "cpu") -> torch.Tensor:
        """The tensor's bytes as stored (U8 as uint8, F8_E4M3 as float8_e4m3fn), optionally
        rows rows.start .. rows.stop - 1 of dimension 0 alone."""
        info = self.header[name]
        dtype = DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        shape = list(info["shape"])
        if rows is not None:
            start, stop, step = rows.indices(shape[0])
            if step != 1:
                raise ValueError("rows must be a contiguous slice")
            per = (end - begin) // shape[0]
            begin, end = begin + start * per, begin + stop * per
            shape[0] = stop - start
        buf = bytearray(end - begin)
        with open(self.path, "rb") as f:
            f.seek(self.data_start + begin)
            if f.readinto(buf) != len(buf):
                raise OSError(f"{self.path}: short read of {name}")
        t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        return t.view(dtype).reshape(shape).to(device)
