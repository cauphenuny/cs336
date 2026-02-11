import torch
import triton
import triton.language as tl
from jaxtyping import Float
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    constexpr = int
else:
    constexpr = tl.constexpr

TILE_SIZE = 16

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Float[torch.Tensor, "batch seq d_model"],
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
    ):
        if is_causal:
            raise NotImplementedError("Causal attention is not implemented for the Triton kernel.")

        batch_size, seq_len, d_model = query.shape
        if key.shape != (batch_size, seq_len, d_model) or value.shape != (batch_size, seq_len, d_model):
            raise ValueError("Expected key/value to have the same shape as query (batch, seq, d_model).")

        output = torch.empty_like(query)
        logsumexp = torch.empty((batch_size, seq_len), device=query.device, dtype=torch.float32)

        if seq_len % TILE_SIZE != 0:
            raise ValueError(f"Sequence length must be a multiple of {TILE_SIZE} for the Triton kernel.")

        # Launch the Triton kernel
        grid = (seq_len // TILE_SIZE, batch_size)
        scale = 1.0 / (d_model**0.5)
        flash_fwd_kernel[grid](
            query, key, value,
            output, logsumexp,
            query.stride(0), query.stride(1), query.stride(2),
            key.stride(0), key.stride(1), key.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            seq_len, seq_len,
            scale,
            D_MODEL=d_model,
            Q_TILE_SIZE=TILE_SIZE,
            K_TILE_SIZE=TILE_SIZE,
        )

        ctx.save_for_backward(logsumexp, query, key, value, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for FlashAttention.")


@triton.jit
def flash_fwd_kernel(
    query_ptr, key_ptr, value_ptr,
    output_ptr, logsumexp_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    num_queries, num_keys,
    scale,
    D_MODEL: constexpr,
    Q_TILE_SIZE: constexpr,
    K_TILE_SIZE: constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_block_ptr = tl.make_block_ptr(
        query_ptr + batch_index * stride_qb,
        shape=(num_queries, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    key_block_ptr = tl.make_block_ptr(
        key_ptr + batch_index * stride_kb,
        shape=(num_keys, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    value_block_ptr = tl.make_block_ptr(
        value_ptr + batch_index * stride_vb,
        shape=(num_keys, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr + batch_index * stride_ob,
        shape=(num_queries, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    logsumexp_block_ptr = tl.make_block_ptr(
        logsumexp_ptr + batch_index * stride_lb,
        shape=(num_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    query = tl.load(query_block_ptr)  # [Q_TILE_SIZE, D_MODEL]
    output = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)
    max_score = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)
    sum_prob = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)

    for _ in range(tl.cdiv(num_keys, K_TILE_SIZE)):
        key_tile = tl.load(key_block_ptr)
        value_tile = tl.load(value_block_ptr)

        score = tl.dot(query, tl.trans(key_tile)) # (q, d) @ (d, k) -> (q, k)
        score = score * scale
        prev_max_score = max_score
        max_score = tl.maximum(max_score, tl.max(score, axis=1, keep_dims=True))
        weight = tl.exp(prev_max_score - max_score)
        prob = tl.exp(score - max_score)
        sum_prob = weight * sum_prob + tl.sum(prob, axis=1, keep_dims=True)
        output = weight * output + tl.dot(prob.to(tl.float32), value_tile.to(tl.float32))

        key_block_ptr = key_block_ptr.advance((K_TILE_SIZE, 0))
        value_block_ptr = value_block_ptr.advance((K_TILE_SIZE, 0))
    
    output = output / sum_prob
    logsumexp = max_score + tl.log(sum_prob)
    
    output = output.to(output_block_ptr.dtype.element_ty)
    logsumexp = logsumexp.to(logsumexp_block_ptr.dtype.element_ty)

    tl.store(output_block_ptr, output)
    tl.store(logsumexp_block_ptr, tl.view(logsumexp, (Q_TILE_SIZE,)))
