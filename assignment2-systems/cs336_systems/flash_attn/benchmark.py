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

"""
result

|   batch | causal   |   seq_len |   d_model | dtype    | impl    |     tile |   fwd_ms |   bwd_ms |   fwd_bwd_ms | status   |
|---------|----------|-----------|-----------|----------|---------|----------|----------|----------|--------------|----------|
|       1 | True     |       128 |        16 | bfloat16 | pytorch | nan      |   0.0573 |   0.3543 |       0.7680 | ok       |
|       1 | True     |       128 |        16 | bfloat16 | triton  |  16.0000 |   0.0133 |   0.1894 |       0.3348 | ok       |
|       1 | True     |       128 |        16 | float32  | pytorch | nan      |   0.0317 |   0.7188 |       1.3640 | ok       |
|       1 | True     |       128 |        16 | float32  | triton  |  16.0000 |   0.0102 |   0.4577 |       1.2155 | ok       |
|       1 | True     |       128 |        32 | bfloat16 | pytorch | nan      |   0.0850 |   0.8837 |       1.7101 | ok       |
|       1 | True     |       128 |        32 | bfloat16 | triton  |  16.0000 |   0.0113 |   0.3748 |       1.2247 | ok       |
|       1 | True     |       128 |        32 | float32  | pytorch | nan      |   0.0317 |   0.7260 |       1.3844 | ok       |
|       1 | True     |       128 |        32 | float32  | triton  |  16.0000 |   0.0143 |   0.3318 |       1.1510 | ok       |
|       1 | True     |       128 |        64 | bfloat16 | pytorch | nan      |   0.0891 |   0.8858 |       1.7157 | ok       |
|       1 | True     |       128 |        64 | bfloat16 | triton  |  16.0000 |   0.0133 |   0.8914 |       1.3670 | ok       |
|       1 | True     |       128 |        64 | float32  | pytorch | nan      |   0.0328 |   0.4751 |       1.3829 | ok       |
|       1 | True     |       128 |        64 | float32  | triton  |  16.0000 |   0.0133 |   0.3789 |       0.7557 | ok       |
|       1 | True     |       128 |       128 | bfloat16 | pytorch | nan      |   0.0850 |   0.8602 |       1.6855 | ok       |
|       1 | True     |       128 |       128 | bfloat16 | triton  |  16.0000 |   0.0133 |   0.8694 |       1.4285 | ok       |
|       1 | True     |       128 |       128 | float32  | pytorch | nan      |   0.0358 |   0.7076 |       1.3466 | ok       |
|       1 | True     |       128 |       128 | float32  | triton  |  16.0000 |   0.0154 |   0.7977 |       1.2933 | ok       |
|       1 | True     |       256 |        16 | bfloat16 | pytorch | nan      |   0.0783 |   0.8750 |       1.7152 | ok       |
|       1 | True     |       256 |        16 | bfloat16 | triton  |  16.0000 |   0.0154 |   0.3871 |       1.3056 | ok       |
|       1 | True     |       256 |        16 | float32  | pytorch | nan      |   0.0410 |   0.1290 |       0.3820 | ok       |
|       1 | True     |       256 |        16 | float32  | triton  |  16.0000 |   0.0174 |   0.3748 |       1.2646 | ok       |
|       1 | True     |       256 |        32 | bfloat16 | pytorch | nan      |   0.0635 |   0.2099 |       0.4813 | ok       |
|       1 | True     |       256 |        32 | bfloat16 | triton  |  16.0000 |   0.0174 |   0.2693 |       0.3917 | ok       |
|       1 | True     |       256 |        32 | float32  | pytorch | nan      |   0.0338 |   0.2043 |       0.3092 | ok       |
|       1 | True     |       256 |        32 | float32  | triton  |  16.0000 |   0.0184 |   0.2120 |       0.3441 | ok       |
|       1 | True     |       256 |        64 | bfloat16 | pytorch | nan      |   0.0604 |   0.3087 |       1.0168 | ok       |
|       1 | True     |       256 |        64 | bfloat16 | triton  |  16.0000 |   0.0184 |   0.3451 |       0.7209 | ok       |
|       1 | True     |       256 |        64 | float32  | pytorch | nan      |   0.0348 |   0.2867 |       0.5233 | ok       |
|       1 | True     |       256 |        64 | float32  | triton  |  16.0000 |   0.0205 |   0.1751 |       0.2816 | ok       |
|       1 | True     |       256 |       128 | bfloat16 | pytorch | nan      |   0.0584 |   0.4726 |       0.7091 | ok       |
|       1 | True     |       256 |       128 | bfloat16 | triton  |  16.0000 |   0.0205 |   0.6211 |       0.5335 | ok       |
|       1 | True     |       256 |       128 | float32  | pytorch | nan      |   0.0389 |   0.4096 |       0.5202 | ok       |
|       1 | True     |       256 |       128 | float32  | triton  |  16.0000 |   0.0236 |   0.5494 |       0.4803 | ok       |
|       1 | True     |       512 |        16 | bfloat16 | pytorch | nan      |   0.0717 |   0.4219 |       1.1325 | ok       |
|       1 | True     |       512 |        16 | bfloat16 | triton  |  16.0000 |   0.0225 |   0.6482 |       0.5151 | ok       |
|       1 | True     |       512 |        16 | float32  | pytorch | nan      |   0.0399 |   0.7301 |       1.3947 | ok       |
|       1 | True     |       512 |        16 | float32  | triton  |  16.0000 |   0.0225 |   0.8090 |       1.1633 | ok       |
|       1 | True     |       512 |        32 | bfloat16 | pytorch | nan      |   0.0911 |   0.8356 |       1.7265 | ok       |
|       1 | True     |       512 |        32 | bfloat16 | triton  |  16.0000 |   0.0256 |   0.9359 |       1.4213 | ok       |
|       1 | True     |       512 |        32 | float32  | pytorch | nan      |   0.0410 |   0.7281 |       1.3978 | ok       |
|       1 | True     |       512 |        32 | float32  | triton  |  16.0000 |   0.0287 |   0.8694 |       1.3409 | ok       |
|       1 | True     |       512 |        64 | bfloat16 | pytorch | nan      |   0.0860 |   0.9073 |       1.7500 | ok       |
|       1 | True     |       512 |        64 | bfloat16 | triton  |  16.0000 |   0.0287 |   0.9405 |       1.4100 | ok       |
|       1 | True     |       512 |        64 | float32  | pytorch | nan      |   0.0522 |   0.7444 |       1.4152 | ok       |
|       1 | True     |       512 |        64 | float32  | triton  |  16.0000 |   0.0328 |   0.8724 |       1.3558 | ok       |
|       1 | True     |       512 |       128 | bfloat16 | pytorch | nan      |   0.0850 |   0.8888 |       1.7285 | ok       |
|       1 | True     |       512 |       128 | bfloat16 | triton  |  16.0000 |   0.0328 |   0.9380 |       1.4131 | ok       |
|       1 | True     |       512 |       128 | float32  | pytorch | nan      |   0.0543 |   0.7240 |       1.3926 | ok       |
|       1 | True     |       512 |       128 | float32  | triton  |  16.0000 |   0.0389 |   0.8709 |       1.3435 | ok       |
|       1 | True     |      1024 |        16 | bfloat16 | pytorch | nan      |   0.0911 |   0.9103 |       1.7536 | ok       |
|       1 | True     |      1024 |        16 | bfloat16 | triton  |  32.0000 |   0.0246 |   0.9523 |       1.4228 | ok       |
|       1 | True     |      1024 |        16 | float32  | pytorch | nan      |   0.0594 |   0.7537 |       1.4264 | ok       |
|       1 | True     |      1024 |        16 | float32  | triton  |  32.0000 |   0.0266 |   0.8120 |       1.3655 | ok       |
|       1 | True     |      1024 |        32 | bfloat16 | pytorch | nan      |   0.0850 |   0.8970 |       1.7357 | ok       |
|       1 | True     |      1024 |        32 | bfloat16 | triton  |  32.0000 |   0.0297 |   0.9472 |       1.4141 | ok       |
|       1 | True     |      1024 |        32 | float32  | pytorch | nan      |   0.0614 |   0.7373 |       1.4029 | ok       |
|       1 | True     |      1024 |        32 | float32  | triton  |  32.0000 |   0.0348 |   0.8141 |       1.3573 | ok       |
|       1 | True     |      1024 |        64 | bfloat16 | pytorch | nan      |   0.0922 |   0.9083 |       1.7490 | ok       |
|       1 | True     |      1024 |        64 | bfloat16 | triton  |  32.0000 |   0.0389 |   0.9554 |       1.4234 | ok       |
|       1 | True     |      1024 |        64 | float32  | pytorch | nan      |   0.0717 |   0.6799 |       1.4203 | ok       |
|       1 | True     |      1024 |        64 | float32  | triton  |  32.0000 |   0.0440 |   0.3922 |       0.7721 | ok       |
|       1 | True     |      1024 |       128 | bfloat16 | pytorch | nan      |   0.1157 |   0.9083 |       1.7454 | ok       |
|       1 | True     |      1024 |       128 | bfloat16 | triton  |  32.0000 |   0.0481 |   0.9508 |       1.4228 | ok       |
|       1 | True     |      1024 |       128 | float32  | pytorch | nan      |   0.0963 |   0.7455 |       1.4090 | ok       |
|       1 | True     |      1024 |       128 | float32  | triton  |  32.0000 |   0.0563 |   0.8407 |       1.3691 | ok       |
|       1 | True     |      2048 |        16 | bfloat16 | pytorch | nan      |   0.1546 |   0.9093 |       1.7490 | ok       |
|       1 | True     |      2048 |        16 | bfloat16 | triton  |  32.0000 |   0.0420 |   0.9554 |       1.4321 | ok       |
|       1 | True     |      2048 |        16 | float32  | pytorch | nan      |   0.1352 |   0.7444 |       1.4141 | ok       |
|       1 | True     |      2048 |        16 | float32  | triton  |  32.0000 |   0.0451 |   0.8868 |       1.3763 | ok       |
|       1 | True     |      2048 |        32 | bfloat16 | pytorch | nan      |   0.1577 |   0.9093 |       1.7449 | ok       |
|       1 | True     |      2048 |        32 | bfloat16 | triton  |  32.0000 |   0.0522 |   0.9554 |       1.4408 | ok       |
|       1 | True     |      2048 |        32 | float32  | pytorch | nan      |   0.1372 |   0.3963 |       0.9073 | ok       |
|       1 | True     |      2048 |        32 | float32  | triton  |  32.0000 |   0.0625 |   0.6420 |       0.5949 | ok       |
|       1 | True     |      2048 |        64 | bfloat16 | pytorch | nan      |   0.1894 |   0.4792 |       0.6595 | ok       |
|       1 | True     |      2048 |        64 | bfloat16 | triton  |  32.0000 |   0.0707 |   0.7096 |       0.5550 | ok       |
|       1 | True     |      2048 |        64 | float32  | pytorch | nan      |   0.1679 |   0.3983 |       0.5622 | ok       |
|       1 | True     |      2048 |        64 | float32  | triton  |  32.0000 |   0.0819 |   0.5161 |       0.4710 | ok       |
|       1 | True     |      2048 |       128 | bfloat16 | pytorch | nan      |   0.2499 |   0.4608 |       0.8069 | ok       |
|       1 | True     |      2048 |       128 | bfloat16 | triton  |  32.0000 |   0.0891 |   0.6840 |       0.5949 | ok       |
|       1 | True     |      2048 |       128 | float32  | pytorch | nan      |   0.2253 |   0.4424 |       0.6748 | ok       |
|       1 | True     |      2048 |       128 | float32  | triton  |  32.0000 |   0.1034 |   0.5550 |       0.5857 | ok       |
|       1 | True     |      4096 |        16 | bfloat16 | pytorch | nan      |   0.5407 |   0.8233 |       1.3527 | ok       |
|       1 | True     |      4096 |        16 | bfloat16 | triton  |  32.0000 |   0.1004 |   0.6989 |       0.7424 | ok       |
|       1 | True     |      4096 |        16 | float32  | pytorch | nan      |   0.5212 |   0.8090 |       1.3189 | ok       |
|       1 | True     |      4096 |        16 | float32  | triton  |  32.0000 |   0.1157 |   0.9134 |       1.0061 | ok       |
|       1 | True     |      4096 |        32 | bfloat16 | pytorch | nan      |   0.5663 |   0.8479 |       1.4070 | ok       |
|       1 | True     |      4096 |        32 | bfloat16 | triton  |  32.0000 |   0.1311 |   0.7004 |       0.8264 | ok       |
|       1 | True     |      4096 |        32 | float32  | pytorch | nan      |   0.5437 |   0.8325 |       1.3701 | ok       |
|       1 | True     |      4096 |        32 | float32  | triton  |  32.0000 |   0.1618 |   0.6605 |       0.8172 | ok       |
|       1 | True     |      4096 |        64 | bfloat16 | pytorch | nan      |   0.6748 |   1.0885 |       1.7551 | ok       |
|       1 | True     |      4096 |        64 | bfloat16 | triton  |  32.0000 |   0.1720 |   1.0138 |       1.1817 | ok       |
|       1 | True     |      4096 |        64 | float32  | pytorch | nan      |   0.6533 |   1.0721 |       1.7162 | ok       |
|       1 | True     |      4096 |        64 | float32  | triton  |  32.0000 |   0.2099 |   0.9697 |       1.1756 | ok       |
|       1 | True     |      4096 |       128 | bfloat16 | pytorch | nan      |   0.9083 |   1.6159 |       2.5211 | ok       |
|       1 | True     |      4096 |       128 | bfloat16 | triton  |  32.0000 |   0.2376 |   1.6620 |       1.8944 | ok       |
|       1 | True     |      4096 |       128 | float32  | pytorch | nan      |   0.8827 |   1.5985 |       2.4791 | ok       |
|       1 | True     |      4096 |       128 | float32  | triton  |  32.0000 |   0.3850 |   1.6067 |       1.9876 | ok       |
|       1 | True     |      8192 |        16 | bfloat16 | pytorch | nan      |   2.0070 |   3.0863 |       5.0847 | ok       |
|       1 | True     |      8192 |        16 | bfloat16 | triton  |  32.0000 |   0.2560 |   2.3368 |       2.5876 | ok       |
|       1 | True     |      8192 |        16 | float32  | pytorch | nan      |   1.9886 |   3.0669 |       5.0493 | ok       |
|       1 | True     |      8192 |        16 | float32  | triton  |  32.0000 |   0.3041 |   2.2159 |       2.5149 | ok       |
|       1 | True     |      8192 |        32 | bfloat16 | pytorch | nan      |   2.1146 |   3.0648 |       5.1702 | ok       |
|       1 | True     |      8192 |        32 | bfloat16 | triton  |  32.0000 |   0.3410 |   2.5477 |       2.8846 | ok       |
|       1 | True     |      8192 |        32 | float32  | pytorch | nan      |   2.0920 |   3.0474 |       5.1313 | ok       |
|       1 | True     |      8192 |        32 | float32  | triton  |  32.0000 |   0.4229 |   2.4228 |       2.8416 | ok       |
|       1 | True     |      8192 |        64 | bfloat16 | pytorch | nan      |   2.5661 |   4.1923 |       6.7543 | ok       |
|       1 | True     |      8192 |        64 | bfloat16 | triton  |  32.0000 |   0.4403 |   3.8461 |       4.2813 | ok       |
|       1 | True     |      8192 |        64 | float32  | pytorch | nan      |   2.5405 |   4.1728 |       6.7103 | ok       |
|       1 | True     |      8192 |        64 | float32  | triton  |  32.0000 |   0.5448 |   3.7120 |       4.2516 | ok       |
|       1 | True     |      8192 |       128 | bfloat16 | pytorch | nan      |   3.5282 |   5.9884 |       9.5089 | ok       |
|       1 | True     |      8192 |       128 | bfloat16 | triton  |  32.0000 |   0.5980 |   6.1573 |       6.7492 | ok       |
|       1 | True     |      8192 |       128 | float32  | pytorch | nan      |   3.4888 |   5.9607 |       9.4459 | ok       |
|       1 | True     |      8192 |       128 | float32  | triton  |  32.0000 |   1.1325 |   6.0047 |       7.1322 | ok       |
|       1 | True     |     16384 |        16 | bfloat16 | pytorch | nan      |   7.2110 |  12.0730 |      19.3193 | ok       |
|       1 | True     |     16384 |        16 | bfloat16 | triton  |  32.0000 |   0.7752 |   9.2457 |      10.0137 | ok       |
|       1 | True     |     16384 |        16 | float32  | pytorch | nan      |   7.2090 |  12.0852 |      19.2829 | ok       |
|       1 | True     |     16384 |        16 | float32  | triton  |  32.0000 |   0.9318 |   8.7967 |       9.7213 | ok       |
|       1 | True     |     16384 |        32 | bfloat16 | pytorch | nan      |   7.6165 |  12.6853 |      20.2824 | ok       |
|       1 | True     |     16384 |        32 | bfloat16 | triton  |  32.0000 |   1.0404 |  10.1048 |      11.0490 | ok       |
|       1 | True     |     16384 |        32 | float32  | pytorch | nan      |   7.5930 |  12.6566 |      20.2435 | ok       |
|       1 | True     |     16384 |        32 | float32  | triton  |  32.0000 |   1.3916 |   9.5596 |      10.9471 | ok       |
|       1 | True     |     16384 |        64 | bfloat16 | pytorch | nan      |   9.6947 |  16.5345 |      26.2257 | ok       |
|       1 | True     |     16384 |        64 | bfloat16 | triton  |  32.0000 |   1.4961 |  14.8091 |      16.2949 | ok       |
|       1 | True     |     16384 |        64 | float32  | pytorch | nan      |   9.6538 |  16.5043 |      26.1530 | ok       |
|       1 | True     |     16384 |        64 | float32  | triton  |  32.0000 |   1.8606 |  14.3350 |      16.1966 | ok       |
|       1 | True     |     16384 |       128 | bfloat16 | pytorch | nan      |  13.4001 |  24.0261 |      37.4118 | ok       |
|       1 | True     |     16384 |       128 | bfloat16 | triton  |  32.0000 |   2.0511 |  24.1690 |      26.2134 | ok       |
|       1 | True     |     16384 |       128 | float32  | pytorch | nan      |  13.3478 |  23.9913 |      37.3371 | ok       |
|       1 | True     |     16384 |       128 | float32  | triton  |  32.0000 |   3.7386 |  23.6626 |      27.3930 | ok       |
|       1 | True     |     32768 |        16 | bfloat16 | pytorch | nan      |  32.4900 |  48.1669 |      80.6697 | ok       |
|       1 | True     |     32768 |        16 | bfloat16 | triton  |  32.0000 |   2.9164 |  36.8927 |      39.8479 | ok       |
|       1 | True     |     32768 |        16 | float32  | pytorch | nan      |  32.4454 |  48.1367 |      80.5821 | ok       |
|       1 | True     |     32768 |        16 | float32  | triton  |  32.0000 |   3.5210 |  35.0894 |      38.2403 | ok       |
|       1 | True     |     32768 |        32 | bfloat16 | pytorch | nan      |  33.9814 |  50.0378 |      84.0095 | ok       |
|       1 | True     |     32768 |        32 | bfloat16 | triton  |  32.0000 |   3.9619 |  40.3533 |      44.1866 | ok       |
|       1 | True     |     32768 |        32 | float32  | pytorch | nan      |  33.9651 |  50.0081 |      83.9393 | ok       |
|       1 | True     |     32768 |        32 | float32  | triton  |  32.0000 |   5.1569 |  38.4686 |      43.3306 | ok       |
|       1 | True     |     32768 |        64 | bfloat16 | pytorch | nan      |  41.0450 |  64.3512 |     105.3839 | ok       |
|       1 | True     |     32768 |        64 | bfloat16 | triton  |  32.0000 |   5.3903 |  57.4147 |      62.7794 | ok       |
|       1 | True     |     32768 |        64 | float32  | pytorch | nan      |  41.0092 |  64.3082 |     105.2733 | ok       |
|       1 | True     |     32768 |        64 | float32  | triton  |  32.0000 |   7.3472 |  56.1623 |      62.9248 | ok       |
|       1 | True     |     32768 |       128 | bfloat16 | pytorch | nan      |  55.3062 |  93.0115 |     148.2916 | ok       |
|       1 | True     |     32768 |       128 | bfloat16 | triton  |  32.0000 |   8.0026 |  93.4543 |     101.8941 | ok       |
|       1 | True     |     32768 |       128 | float32  | pytorch | nan      |  55.2202 |  92.9459 |     148.1595 | ok       |
|       1 | True     |     32768 |       128 | float32  | triton  |  32.0000 |  14.8900 |  91.5656 |     106.4561 | ok       |
|       1 | True     |     65536 |        16 | bfloat16 | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        16 | bfloat16 | triton  |  32.0000 |  10.7761 | 136.4705 |     147.9813 | ok       |
|       1 | True     |     65536 |        16 | float32  | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        16 | float32  | triton  |  32.0000 |  13.0550 | 137.6911 |     150.3427 | ok       |
|       1 | True     |     65536 |        32 | bfloat16 | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        32 | bfloat16 | triton  |  32.0000 |  14.7067 | 149.4006 |     164.4319 | ok       |
|       1 | True     |     65536 |        32 | float32  | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        32 | float32  | triton  |  32.0000 |  19.6173 | 150.7492 |     170.5093 | ok       |
|       1 | True     |     65536 |        64 | bfloat16 | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        64 | bfloat16 | triton  |  32.0000 |  20.0131 | 221.1256 |     241.5606 | ok       |
|       1 | True     |     65536 |        64 | float32  | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |        64 | float32  | triton  |  32.0000 |  27.0684 | 224.0533 |     251.6122 | ok       |
|       1 | True     |     65536 |       128 | bfloat16 | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |       128 | bfloat16 | triton  |  32.0000 |  29.5977 | 366.8449 |     396.4119 | ok       |
|       1 | True     |     65536 |       128 | float32  | pytorch | nan      | nan      | nan      |     nan      | oom      |
|       1 | True     |     65536 |       128 | float32  | triton  |  32.0000 |  56.5228 | 369.2759 |     426.0905 | ok       |
"""