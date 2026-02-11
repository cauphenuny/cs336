import torch
from einops import rearrange, einsum
from jaxtyping import Float

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
            raise NotImplementedError("Causal attention is not implemented yet.")
        queries = rearrange(query, "b (l t) d -> l b t d", t=TILE_SIZE)
        keys = rearrange(key, "b (l t) d -> l b t d", t=TILE_SIZE)
        values = rearrange(value, "b (l t) d -> l b t d", t=TILE_SIZE)
        _, batch_size, tile_size, dim = queries.shape
        outputs = []
        logsumexps = []

        for q in queries:
            output = torch.zeros_like(q)  # b t d
            max_score = torch.full((batch_size, tile_size, 1), float("-inf"), device=q.device)
            sum_prob = torch.zeros((batch_size, tile_size, 1), device=q.device)

            for k, v in zip(keys, values):
                score: Float[torch.Tensor, "b t t"] = einsum(
                    q,
                    k,
                    "b tq d, b tk d -> b tq tk",
                ) / (dim**0.5)
                prev_max_score = max_score
                max_score = torch.maximum(max_score, torch.max(score, dim=-1, keepdim=True).values)
                weight = torch.exp(prev_max_score - max_score)
                prob = torch.exp(score - max_score)
                sum_prob = weight * sum_prob + torch.sum(prob, dim=-1, keepdim=True)
                output = weight * output + prob @ v

            output = (1 / sum_prob) * output
            logsumexp = max_score + torch.log(sum_prob)
            outputs.append(output)
            logsumexps.append(logsumexp.squeeze(-1))

        output: Float[torch.Tensor, "batch len dim"] = torch.cat(outputs, dim=-2)
        logsumexp: Float[torch.Tensor, "batch len"] = torch.cat(logsumexps, dim=-1)

        ctx.save_for_backward(logsumexp, query, key, value, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for FlashAttention.")


f_flashattn = FlashAttention.apply
