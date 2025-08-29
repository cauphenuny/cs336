def cost(
    name: str, batch_size, seq_len, d_model, d_ff, num_heads, num_layers, vocab_size
):
    """
    1. Embeddings: $0$
    2. TransformerBlock:
        1. `qkv_proj`: $3\\times 2\\times B \\times L\\times D^2=6BLD^2$
        2. `attn`: $2 \\times H\\times B \\times L^2 \\times \\dfrac{D}{H}+2\\times H \\times B \\times L^2\\times \\dfrac{D}{H}=4BDL^2$
        3. `o_proj`: $2 \\times B\\times L\\times D^2=2BLD^2$
        4. `ffn`: $2\\times B\\times L\\times D\\times D_{ff}\\times 2+2\\times B\\times L\\times D\\times {D_{ff}}=6BLD D_{ff}$
        sum: $((8BLD^2+4BDL^2)+6 BLD D_{ff})\\times N$
    3. output: $2BLDV$
    """
    if not d_ff:
        d_ff = d_model * 4

    qkv_proj = 6 * batch_size * seq_len * d_model**2
    product = 4 * batch_size * seq_len**2 * d_model
    o_proj = 2 * batch_size * seq_len * d_model**2
    attn = (qkv_proj + o_proj + product) * num_layers
    ffn = (6 * batch_size * seq_len * d_model * d_ff) * num_layers

    output = 2 * batch_size * seq_len * d_model * vocab_size
    all = attn + ffn + output
    print(f"{name} | {all} | ", end="")
    print(f"attn: {attn}/{attn / all * 100:.2f}%", end="\t")
    print(f"ffn: {ffn}/{ffn / all * 100:.2f}%", end="\t")
    print(f"output: {output}/{output / all * 100:.2f}%")

    return attn + ffn + output


if __name__ == "__main__":
    cost(
        "GPT2-XL",
        batch_size=1,
        seq_len=1024,
        d_model=1600,
        d_ff=None,
        num_heads=25,
        num_layers=48,
        vocab_size=50257,
    )
    cost(
        "GPT2-Large",
        batch_size=1,
        seq_len=1024,
        d_model=1280,
        d_ff=None,
        num_heads=20,
        num_layers=36,
        vocab_size=50257,
    )
    cost(
        "GPT2-Medium",
        batch_size=1,
        seq_len=1024,
        d_model=1024,
        d_ff=None,
        num_heads=16,
        num_layers=24,
        vocab_size=50257,
    )
    cost(
        "GPT2-Small",
        batch_size=1,
        seq_len=1024,
        d_model=768,
        d_ff=None,
        num_heads=12,
        num_layers=12,
        vocab_size=50257,
    )
    cost(
        "GPT2-XL-large-context",
        batch_size=1,
        seq_len=16384,
        d_model=1600,
        d_ff=None,
        num_heads=25,
        num_layers=48,
        vocab_size=50257,
    )
