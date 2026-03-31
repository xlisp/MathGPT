"""
PyTorch 2.3.x / Python 3.12 兼容补丁。

必须在导入任何 nanochat 模块之前调用 apply()。
修复三个问题：
  1. F.rms_norm          — PyTorch 2.4 才加入，2.3 没有
  2. torch.compile       — Python 3.12 + PyTorch 2.3 的 Dynamo 不支持，改为 no-op
  3. enable_gqa in SDPA  — PyTorch 2.5 才加入，用 repeat_interleave 手动展开
"""

import torch
import torch.nn.functional as F


def apply():
    _patch_rms_norm()
    _patch_torch_compile()
    _patch_sdpa_enable_gqa()


# ── 补丁 1：F.rms_norm ────────────────────────────────────────────
def _patch_rms_norm():
    if hasattr(F, "rms_norm"):
        return  # 已有，无需补丁

    def rms_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        """RMS Norm，等价于 PyTorch 2.4+ 的 F.rms_norm。"""
        # 在 float32 下计算方差，避免 fp16 溢出
        dims = tuple(range(-len(normalized_shape), 0))
        variance = input.float().pow(2).mean(dims, keepdim=True)
        output = input * torch.rsqrt(variance + eps)
        # 保持输入 dtype
        output = output.to(input.dtype)
        if weight is not None:
            output = output * weight
        if bias is not None:
            output = output + bias
        return output

    F.rms_norm = rms_norm
    print("[compat] Patched F.rms_norm (PyTorch < 2.4)")


# ── 补丁 2：torch.compile no-op ───────────────────────────────────
def _patch_torch_compile():
    """Python 3.12 + PyTorch 2.3 的 Dynamo 不可用，改为恒等装饰器。"""
    try:
        @torch.compile(dynamic=False, fullgraph=True)
        def _test(x):
            return x + 1
        _test(torch.tensor(1))  # 实际触发编译
        # 如果没有抛异常，说明本环境支持 compile，不需要补丁
        return
    except Exception:
        pass

    _original_compile = torch.compile  # noqa: F841

    def _noop_compile(*args, **kwargs):
        """torch.compile 的 no-op 版本：直接返回函数本身。"""
        # 情况 A：@torch.compile  (fn 作为第一个位置参数)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # 情况 B：@torch.compile(...)  (带参数，返回装饰器)
        return lambda fn: fn

    torch.compile = _noop_compile
    print("[compat] Patched torch.compile → no-op (Python 3.12 + PyTorch < 2.4)")


# ── 补丁 3：F.scaled_dot_product_attention enable_gqa ────────────
def _patch_sdpa_enable_gqa():
    """PyTorch 2.3 的 SDPA 不接受 enable_gqa 参数，手动展开 KV heads。"""
    # 用实际调用探测是否支持 enable_gqa（C 扩展无法用 inspect.signature）
    try:
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 1, 1, 4)
        v = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return  # 支持，无需补丁
    except TypeError:
        pass  # 不支持，继续打补丁
    except Exception:
        pass  # 其他错误（如形状问题），继续打补丁

    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_with_gqa(query, key, value,
                       attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None,
                       enable_gqa=False, **kwargs):
        if enable_gqa:
            # query: (B, H_q, T, D)  key/value: (B, H_kv, T, D)
            h_q  = query.size(-3)
            h_kv = key.size(-3)
            if h_q != h_kv:
                assert h_q % h_kv == 0, f"H_q ({h_q}) must be divisible by H_kv ({h_kv})"
                n_rep = h_q // h_kv
                key   = key.repeat_interleave(n_rep, dim=-3)
                value = value.repeat_interleave(n_rep, dim=-3)
        return _orig_sdpa(query, key, value,
                          attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = _sdpa_with_gqa
    print("[compat] Patched F.scaled_dot_product_attention → enable_gqa support (PyTorch < 2.5)")
