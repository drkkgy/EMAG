"""
Unit tests for the EMAG core (attention processor + emag_forward), CPU-only.

These test the paper's contribution in isolation with a mock SD3-style joint-attention
module — no SD3 weights, no GPU, no network. Run with:

    pytest tests/                      # if pytest is installed
    python tests/test_emag_attention.py   # plain-python fallback
"""
import torch
import torch.nn as nn

from emag.emag_util import EMAGAttnProcessor2_0, emag_forward, k2idx
from diffusers.models.attention_processor import JointAttnProcessor2_0

torch.manual_seed(0)


class MockJointAttention(nn.Module):
    """Minimal stand-in for a diffusers SD3 joint-attention `Attention` module.

    Exposes exactly the attributes both JointAttnProcessor2_0 and EMAGAttnProcessor2_0
    read, so the two can be compared on identical weights.
    """

    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(dim, dim)
        self.context_pre_only = False


def _inputs(B=2, N_img=16, N_txt=8, dim=64):
    hidden = torch.randn(B, N_img, dim)
    enc = torch.randn(B, N_txt, dim)
    return hidden, enc


# ---------------------------------------------------------------------------


def test_k2idx_mapping():
    # k in [0, k_base] maps linearly onto [0, N-1]
    assert k2idx(0, 28) == 0
    assert k2idx(250, 28) == 27          # k_base -> last index
    assert k2idx(125, 28) == 14          # 125*27/250 = 13.5 -> round() gives 14
    assert k2idx(10_000, 28) == 27       # clamped high
    assert k2idx(-5, 28) == 0            # clamped low
    print("PASS test_k2idx_mapping")


def test_emag_disabled_equals_standard_joint_attention():
    """With EMAG disabled, the processor must reproduce diffusers' JointAttnProcessor2_0."""
    attn = MockJointAttention().eval()
    hidden, enc = _inputs()

    ref = JointAttnProcessor2_0()
    out_ref = ref(attn, hidden.clone(), encoder_hidden_states=enc.clone())

    proc = EMAGAttnProcessor2_0()
    proc.enable_emag = False
    out_emag = proc(attn, hidden.clone(), encoder_hidden_states=enc.clone())

    # both return (image_out, ctx_out)
    assert isinstance(out_ref, tuple) and isinstance(out_emag, tuple)
    for a, b in zip(out_emag, out_ref):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-4), f"max diff {(a - b).abs().max().item()}"
    print("PASS test_emag_disabled_equals_standard_joint_attention")


def test_shapes_preserved_with_emag():
    attn = MockJointAttention().eval()
    hidden, enc = _inputs()
    proc = EMAGAttnProcessor2_0()
    proc.attn_ema_full = True
    img_out, ctx_out = proc(attn, hidden.clone(), encoder_hidden_states=enc.clone())
    assert img_out.shape == hidden.shape
    assert ctx_out.shape == enc.shape
    print("PASS test_shapes_preserved_with_emag")


def test_ema_accumulation_populates_state():
    attn = MockJointAttention().eval()
    hidden, enc = _inputs()
    proc = EMAGAttnProcessor2_0()
    proc.attn_ema_full = True
    proc.do_ema = True
    assert proc.attn_ema is None
    proc(attn, hidden.clone(), encoder_hidden_states=enc.clone())
    assert proc.attn_ema is not None, "EMA accumulator should be populated after do_ema"
    assert proc.attn_gradient is not None, "attention gradient should be computed"
    print("PASS test_ema_accumulation_populates_state")


def test_emag_q_disabled_equals_standard_joint_attention():
    """EMAG-Q with no accumulate/apply uses fused SDPA and must match standard joint attention."""
    attn = MockJointAttention().eval()
    hidden, enc = _inputs()

    ref = JointAttnProcessor2_0()
    out_ref = ref(attn, hidden.clone(), encoder_hidden_states=enc.clone())

    proc = EMAGAttnProcessor2_0(use_q_ema=True)
    proc.use_q_ema = True  # SDPA query-EMA path
    out_q = proc(attn, hidden.clone(), encoder_hidden_states=enc.clone())

    for a, b in zip(out_q, out_ref):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-4), f"max diff {(a - b).abs().max().item()}"
    print("PASS test_emag_q_disabled_equals_standard_joint_attention")


def test_emag_q_accumulation_populates_state():
    attn = MockJointAttention().eval()
    hidden, enc = _inputs()
    proc = EMAGAttnProcessor2_0(use_q_ema=True)
    proc.use_q_ema = True
    proc.do_ema = True
    assert proc.q_ema is None
    proc(attn, hidden.clone(), encoder_hidden_states=enc.clone())
    assert proc.q_ema is not None, "query-EMA accumulator should be populated after do_ema"
    assert proc.attn_gradient is not None, "attn_gradient (MAE of q vs q_ema) should be computed"
    print("PASS test_emag_q_accumulation_populates_state")


def test_emag_q_application_changes_output():
    """Accumulate q-EMA on input A, apply to input B -> output shifts vs baseline."""
    attn = MockJointAttention().eval()
    hidden_a, enc_a = _inputs()
    hidden_b, enc_b = _inputs()

    proc = EMAGAttnProcessor2_0(use_q_ema=True)
    proc.use_q_ema = True

    proc.do_ema = True
    proc.update_ema_to_attn = False
    proc(attn, hidden_a.clone(), encoder_hidden_states=enc_a.clone())

    proc.do_ema = False
    proc.update_ema_to_attn = False
    base_img, _ = proc(attn, hidden_b.clone(), encoder_hidden_states=enc_b.clone())

    proc.update_ema_to_attn = True
    q_img, _ = proc(attn, hidden_b.clone(), encoder_hidden_states=enc_b.clone())

    diff = (q_img - base_img).abs().max().item()
    assert diff > 1e-4, f"query-EMA application should change the output (max diff {diff})"
    print(f"PASS test_emag_q_application_changes_output (max diff {diff:.5f})")


def test_ema_application_changes_output():
    """Accumulate an EMA on one input, then apply it to a different input -> output shifts."""
    attn = MockJointAttention().eval()
    hidden_a, enc_a = _inputs()
    hidden_b, enc_b = _inputs(N_img=16, N_txt=8)  # different random content

    proc = EMAGAttnProcessor2_0()
    proc.attn_ema_full = True

    # 1) accumulate EMA from input A
    proc.do_ema = True
    proc.update_ema_to_attn = False
    proc(attn, hidden_a.clone(), encoder_hidden_states=enc_a.clone())

    # 2) baseline (no apply) on input B
    proc.do_ema = False
    proc.update_ema_to_attn = False
    base_img, _ = proc(attn, hidden_b.clone(), encoder_hidden_states=enc_b.clone())

    # 3) apply EMA on input B
    proc.update_ema_to_attn = True
    emag_img, _ = proc(attn, hidden_b.clone(), encoder_hidden_states=enc_b.clone())

    diff = (emag_img - base_img).abs().max().item()
    assert diff > 1e-4, f"EMA application should change the output (max diff {diff})"
    print(f"PASS test_ema_application_changes_output (max diff {diff:.5f})")


# --- emag_forward: windowing + adaptive layer selection --------------------


class _MockAttn(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor


class _MockBlock(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.attn = _MockAttn(processor)


class _MockTransformer(nn.Module):
    """Records each processor's toggle state at call time."""

    def __init__(self, n_layers=12):
        super().__init__()
        blocks = []
        for _ in range(n_layers):
            p = EMAGAttnProcessor2_0()
            blocks.append(_MockBlock(p))
        self.transformer_blocks = nn.ModuleList(blocks)
        self.captured = None

    def forward(self, **kwargs):
        self.captured = {
            i: (blk.attn.processor.do_ema, blk.attn.processor.update_ema_to_attn)
            for i, blk in enumerate(self.transformer_blocks)
        }
        return (torch.zeros(1),)


def test_emag_forward_windows_and_reset():
    tf = _MockTransformer(n_layers=12)
    layers = [6, 7, 8]
    # give layer 7 the largest attention gradient so adaptive mode 2 picks it
    for i, g in zip(layers, [0.1, 0.9, 0.3]):
        tf.transformer_blocks[i].attn.processor.attn_gradient = torch.tensor(g)

    # step inside both the accumulate and apply windows
    out = emag_forward(
        tf, hidden_states=None, timestep=None, encoder_hidden_states=None,
        pooled_projections=None, joint_attention_kwargs=None, return_dict=False,
        emag_start_step=8, emag_stop_step=1, emag_time_delta=2,
        emag_adaptive_mode=2, emag_layers=layers, step_index=6,
    )
    assert isinstance(out, tuple)

    cap = tf.captured
    # accumulate (do_ema) toggled on ALL candidate layers during the call
    for i in layers:
        assert cap[i][0] is True, f"layer {i} should have do_ema=True during forward"
    # apply (update_ema_to_attn) toggled only on the highest-gradient layer (7)
    assert cap[7][1] is True, "highest-gradient layer 7 should be selected for EMA apply"
    assert cap[6][1] is False and cap[8][1] is False, "non-selected layers must not apply"

    # all flags reset AFTER the forward
    for i in layers:
        assert tf.transformer_blocks[i].attn.processor.do_ema is False
        assert tf.transformer_blocks[i].attn.processor.update_ema_to_attn is False
    print("PASS test_emag_forward_windows_and_reset")


def test_emag_forward_outside_window_no_apply():
    tf = _MockTransformer(n_layers=12)
    layers = [6, 7, 8]
    for i in layers:
        tf.transformer_blocks[i].attn.processor.attn_gradient = torch.tensor(0.5)
    # step_index above start -> neither accumulate nor apply
    emag_forward(
        tf, hidden_states=None, timestep=None, encoder_hidden_states=None,
        pooled_projections=None, joint_attention_kwargs=None, return_dict=False,
        emag_start_step=8, emag_stop_step=1, emag_time_delta=2,
        emag_adaptive_mode=2, emag_layers=layers, step_index=10,
    )
    for i in layers:
        assert tf.captured[i] == (False, False)
    print("PASS test_emag_forward_outside_window_no_apply")


ALL_TESTS = [
    test_k2idx_mapping,
    test_emag_disabled_equals_standard_joint_attention,
    test_shapes_preserved_with_emag,
    test_ema_accumulation_populates_state,
    test_ema_application_changes_output,
    test_emag_q_disabled_equals_standard_joint_attention,
    test_emag_q_accumulation_populates_state,
    test_emag_q_application_changes_output,
    test_emag_forward_windows_and_reset,
    test_emag_forward_outside_window_no_apply,
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
