"""Benchmark Triton FlashAttention-2 (partial) vs naive PyTorch attention.

This script uses triton.testing.do_bench to measure:
- forward latency
- backward-only latency (reuse graph via retain_graph=True)
- end-to-end forward+backward latency

Constraints required:
- batch size is always 1
- causal masking always enabled
- sweep S in powers of 2 from 128..65536
- sweep D in powers of 2 from 16..128
- sweep dtypes: bfloat16 and float32
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd
import torch
import triton.testing
from tabulate import tabulate

from . import triton_kernel


ImplName = Literal["triton", "pytorch"]


def _device() -> torch.device:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required to benchmark Triton kernels, but torch.cuda.is_available() is False.")
	return torch.device("cuda")


def _causal_attention_pytorch(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	*,
	causal_mask: torch.Tensor,
) -> torch.Tensor:
	"""Naive PyTorch attention: (q @ k^T) -> softmax -> @ v, with causal masking.

	Notes:
	- This intentionally materializes the full (S,S) score matrix, so it may OOM for large S.
	- We compute softmax in fp32 for numerical stability, then cast output back to input dtype.
	"""

	# q,k,v: [B, S, D] with B==1
	b, s, d = q.shape
	assert b == 1
	scale = 1.0 / math.sqrt(d)

	qf = q.float()
	kf = k.float()
	vf = v.float()

	scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale  # [1, S, S]

	# Causal mask: disallow attending to future keys.
	# Use a large negative finite value (like the provided reference/tests) instead of -inf.
	scores = scores.masked_fill(causal_mask, -1.0e6)

	probs = torch.softmax(scores, dim=-1)
	out = torch.matmul(probs, vf)  # [1, S, D]
	return out.to(dtype=q.dtype)


def _pick_tile_size(seq_len: int, d_model: int) -> int:
	"""Heuristic tile-size selection for the Triton kernel.

	This codebase's Triton kernel requires seq_len % TILE_SIZE == 0.
	All seq_len we benchmark are powers of 2, so any power-of-2 tile <= seq_len works.
	"""

	if seq_len >= 16384:
		# return 128
		return 32
	if seq_len >= 4096:
		# return 64
		return 32
	if seq_len >= 1024:
		return 32
	return 16


@dataclass(frozen=True)
class BenchConfig:
	warmup_ms: int = 50
	rep_ms: int = 200
	return_mode: str = "median"


def _bench_ms(cfg: BenchConfig, fn: Callable[[], None]) -> float:
	# Ensure no stray work leaks across measurements.
	torch.cuda.synchronize()
	result = triton.testing.do_bench(
		fn,
		warmup=cfg.warmup_ms,
		rep=cfg.rep_ms,
		return_mode=cfg.return_mode,
		quantiles=None,
	)
	# do_bench can return a float or a tuple/list depending on options.
	if isinstance(result, (tuple, list)):
		if len(result) == 0:
			raise RuntimeError("triton.testing.do_bench returned an empty result")
		result = result[0]
	if result is None:
		raise RuntimeError("triton.testing.do_bench returned None")
	return float(result)


def _run_one(
	*,
	impl: ImplName,
	q0: torch.Tensor,
	k0: torch.Tensor,
	v0: torch.Tensor,
	grad_out0: torch.Tensor,
	tile_size: int | None,
	cfg: BenchConfig,
) -> dict:
	"""Benchmark forward, backward-only, and forward+backward for one implementation."""

	q = q0.clone().detach().requires_grad_(True)
	k = k0.clone().detach().requires_grad_(True)
	v = v0.clone().detach().requires_grad_(True)
	grad_out = grad_out0.clone().detach()

	causal_mask = None
	if impl == "pytorch":
		# Precompute once; avoid including mask construction in forward timing.
		s = q.shape[1]
		causal_mask = torch.triu(
			torch.ones((s, s), device=q.device, dtype=torch.bool),
			diagonal=1,
		)

	tile_guard = None
	if impl == "triton":
		assert tile_size is not None
		tile_guard = triton_kernel.TILE_SIZE
		triton_kernel.TILE_SIZE = tile_size

	try:
		def forward() -> torch.Tensor:
			if impl == "triton":
				return triton_kernel.f_flashattn(q, k, v, True)
			if impl == "pytorch":
				assert causal_mask is not None
				return _causal_attention_pytorch(q, k, v, causal_mask=causal_mask)
			raise AssertionError(f"Unknown impl: {impl}")

		# Warm up compilation / caching.
		_ = forward()
		torch.cuda.synchronize()

		def fwd_only() -> None:
			_ = forward()

		fwd_ms = _bench_ms(cfg, fwd_only)

		# Backward-only benchmark: reuse the same graph.
		out = forward()
		torch.cuda.synchronize()

		def bwd_only() -> None:
			q.grad = None
			k.grad = None
			v.grad = None
			out.backward(grad_out, retain_graph=True)

		bwd_ms = _bench_ms(cfg, bwd_only)

		def fwd_bwd() -> None:
			q.grad = None
			k.grad = None
			v.grad = None
			o = forward()
			o.backward(grad_out)

		fwd_bwd_ms = _bench_ms(cfg, fwd_bwd)
	finally:
		if impl == "triton" and tile_guard is not None:
			triton_kernel.TILE_SIZE = tile_guard

	return {
		"impl": impl,
		"tile": tile_size if impl == "triton" else None,
		"fwd_ms": fwd_ms,
		"bwd_ms": bwd_ms,
		"fwd_bwd_ms": fwd_bwd_ms,
		"status": "ok",
	}


def main() -> None:
	device = _device()
	batch_size = 1
	is_causal = True

	seq_lens = [2**p for p in range(7, 17)]  # 128..65536
	d_models = [2**p for p in range(4, 8)]  # 16..128
	dtypes = [torch.bfloat16, torch.float32]

	cfg = BenchConfig()
	rows: list[dict] = []

	# Keep the printed table stable/readable.
	torch.set_grad_enabled(True)
	torch.backends.cudnn.benchmark = True

	for seq_len in seq_lens:
		for d_model in d_models:
			for dtype in dtypes:
				# Generate inputs once per (S, D, dtype), before benchmarking.
				# Use a fixed distribution but not a fixed seed (prompt only requires randomness).
				q0 = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)
				k0 = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)
				v0 = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)
				grad_out0 = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)

				tile = _pick_tile_size(seq_len, d_model)
				assert seq_len % tile == 0

				for impl in ("triton", "pytorch"):
					base = {
						"batch": batch_size,
						"causal": is_causal,
						"seq_len": seq_len,
						"d_model": d_model,
						"dtype": str(dtype).replace("torch.", ""),
					}
					try:
						result = _run_one(
							impl=impl,
							q0=q0,
							k0=k0,
							v0=v0,
							grad_out0=grad_out0,
							tile_size=tile,
							cfg=cfg,
						)
						rows.append({**base, **result})
					except torch.cuda.OutOfMemoryError:
						torch.cuda.empty_cache()
						rows.append(
							{
								**base,
								"impl": impl,
								"tile": tile if impl == "triton" else None,
								"fwd_ms": float("nan"),
								"bwd_ms": float("nan"),
								"fwd_bwd_ms": float("nan"),
								"status": "oom",
							}
						)
					except RuntimeError as e:
						# Catch common OOM variants that manifest as RuntimeError.
						if "out of memory" in str(e).lower():
							torch.cuda.empty_cache()
							rows.append(
								{
									**base,
									"impl": impl,
									"tile": tile if impl == "triton" else None,
									"fwd_ms": float("nan"),
									"bwd_ms": float("nan"),
									"fwd_bwd_ms": float("nan"),
									"status": "oom",
								}
							)
						else:
							rows.append(
								{
									**base,
									"impl": impl,
									"tile": tile if impl == "triton" else None,
									"fwd_ms": float("nan"),
									"bwd_ms": float("nan"),
									"fwd_bwd_ms": float("nan"),
									"status": f"error: {type(e).__name__}",
								}
							)

	df = pd.DataFrame(rows)
	df = df.sort_values(["seq_len", "d_model", "dtype", "impl"], kind="stable")

	# Print as a readable table.
	print(
		tabulate(
			df,
			headers="keys",
			tablefmt="github",
			showindex=False,
			floatfmt=".4f",
		)
	)


if __name__ == "__main__":
	main()

