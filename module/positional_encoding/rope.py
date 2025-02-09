import torch
import torch.nn as nn


def precompute_freqs_cis(embed_size, max_len, theta=10000):
    freqs = 1.0 / (
        theta ** torch.arange(0, embed_size, 2)[: embed_size // 2].float() / embed_size
    )
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    # 创建模长为1，shape为（end, embed_size//2）复数张量
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    xq = torch.view_as_real(xq * freqs_cis).flatten(3)
    xk = torch.view_as_real(xk * freqs_cis).flatten(3)

    return xq, xk


if __name__ == "__main__":
    embed_size = 64
    max_length = 512
    freqs_cis = precompute_freqs_cis(embed_size, max_length)

    batch_size = 32
    seq_length = 512
    num_heads = 16
    head_dim = 64

    xq = torch.randn(32, num_heads, seq_length, head_dim)
    xk = torch.randn(32, num_heads, seq_length, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    print(xq.shape, xk.shape)
