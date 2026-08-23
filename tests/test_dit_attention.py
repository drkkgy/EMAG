"""
Unit tests for the DiT-side EMAG custom attention (class-conditional path), CPU-only.

No DiT weights or GPU needed. Run with:
    pytest tests/
    python tests/test_dit_attention.py
"""
import os
import sys

import torch

# make the flat-layout DiT code importable (dit/custom_code/...)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dit"))

from custom_code.custom_attention import Attention  # noqa: E402

torch.manual_seed(0)


def _module(dim=64, heads=4):
    m = Attention(dim=dim, num_heads=heads, qkv_bias=True)
    return m.eval()


def test_manual_attention_matches_sdpa():
    """The manual (attn-exposing) path must equal fused scaled_dot_product_attention."""
    m = _module()
    x = torch.randn(2, 16, 64)

    m.enable_emag = False
    m.fused_attn = False
    out_manual = m(x.clone())

    m.fused_attn = True  # switch to torch SDPA on identical weights
    out_fused = m(x.clone())

    assert out_manual.shape == x.shape
    diff = (out_manual - out_fused).abs().max().item()
    assert diff < 1e-4, f"manual vs SDPA mismatch (max diff {diff})"
    print(f"PASS test_manual_attention_matches_sdpa (max diff {diff:.6f})")


def test_dit_ema_accumulation_populates_state():
    m = _module()
    x = torch.randn(2, 16, 64)
    m.enable_emag = True
    m.do_ema = True
    assert m.attn_ema is None
    m(x.clone())
    assert m.attn_ema is not None, "EMA accumulator should be set after do_ema"
    assert m.attn_entropy is not None, "attention entropy should be computed"
    assert m.attn_gradient is not None, "attention gradient should be computed"
    print("PASS test_dit_ema_accumulation_populates_state")


def test_dit_ema_application_changes_output():
    m = _module()
    xa = torch.randn(2, 16, 64)
    xb = torch.randn(2, 16, 64)

    m.enable_emag = True
    # accumulate EMA from input A
    m.do_ema = True
    m.update_ema_to_attn = False
    m(xa.clone())

    # baseline on B (no apply)
    m.do_ema = False
    m.update_ema_to_attn = False
    base = m(xb.clone())

    # apply EMA on B
    m.update_ema_to_attn = True
    emag = m(xb.clone())

    diff = (emag - base).abs().max().item()
    assert diff > 1e-4, f"EMA application should change the output (max diff {diff})"
    print(f"PASS test_dit_ema_application_changes_output (max diff {diff:.6f})")


ALL_TESTS = [
    test_manual_attention_matches_sdpa,
    test_dit_ema_accumulation_populates_state,
    test_dit_ema_application_changes_output,
]


if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tests passed")
    raise SystemExit(1 if failures else 0)
