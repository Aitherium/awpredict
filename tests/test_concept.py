"""ConceptWorldModel: learns concept dynamics, does not collapse, ranks drafts."""
import importlib.util

import pytest
from awpredict.contracts import WorldModel, conforms

_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

D_IN, K, N_PROTO = 16, 4, 6


def _world(seed=0):
    """Concept prototypes whose order follows a fixed cycle: 0->1->...->5->0."""
    import torch
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(N_PROTO, D_IN, generator=g) * 2.0

    def seq(start, n_chunks, order=None, noise=0.3):
        ids = order or [(start + i) % N_PROTO for i in range(n_chunks)]
        rows = [protos[i] + noise * torch.randn(K, D_IN, generator=g) for i in ids]
        return torch.cat(rows, 0), ids

    return protos, seq


@pytest.fixture(scope="module")
def trained():
    import torch
    from awpredict.core.concept import ConceptWorldModel
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))   # 32 threads on a busy host: 25x slower
    _, seq = _world()
    m = ConceptWorldModel(d_in=D_IN, dim=64, k=K, segments=4, codes=16, levels=2,
                          layers=1, heads=4, max_concepts=32)
    for i in range(256):
        m.observe(seq(i % N_PROTO, 8)[0])
    log = m.train_step(batch_size=16, steps=300)
    return m, seq, log


@needs_torch
def test_conforms_to_world_model_contract():
    from awpredict.core.concept import ConceptWorldModel
    m = ConceptWorldModel(d_in=D_IN)
    assert conforms(m, WorldModel) == []
    assert m.ok


@needs_torch
def test_learns_the_concept_order(trained):
    import torch
    m, seq, _ = trained
    in_order = sum(m.surprise(seq(s, 8)[0]) for s in range(N_PROTO))
    g = torch.Generator().manual_seed(7)
    shuffled = 0.0
    for s in range(N_PROTO):
        order = torch.randperm(N_PROTO, generator=g).tolist() + [s, (s + 3) % N_PROTO]
        shuffled += m.surprise(seq(0, 8, order=order)[0])
    assert shuffled > 3 * in_order, (in_order, shuffled)


@needs_torch
def test_no_collapse(trained):
    m, seq, log = trained
    z = m.encode(seq(0, 8)[0])
    assert float(z.std(0).mean()) > 0.1
    assert log["sigreg"] < 0.1
    ids = [m.concept_ids(seq(s, 8)[0]) for s in range(N_PROTO)]
    coarse = {tuple(chunk[0]) for row in ids for chunk in row}
    assert len(coarse) >= N_PROTO - 1        # distinct concepts get distinct coarse codes


@needs_torch
def test_plan_picks_the_true_continuation(trained):
    m, seq, _ = trained
    hits = 0
    for s in range(N_PROTO):
        ctx, _ = seq(s, 5)
        nxt = (s + 5) % N_PROTO
        cands = [seq(0, 1, order=[j])[0] for j in range(N_PROTO)]
        hits += m.plan(ctx, cands)[0] == nxt
    assert hits >= N_PROTO - 1


@needs_torch
def test_save_load_roundtrip(trained, tmp_path):
    import torch
    from awpredict.core.concept import ConceptWorldModel
    m, seq, _ = trained
    p = str(tmp_path / "c.ckpt")
    assert m.save(p)
    m2 = ConceptWorldModel(**m.cfg)
    assert m2.load(p)
    x = seq(2, 6)[0]
    assert torch.allclose(m.encode(x), m2.encode(x))
    assert not m2.load(str(tmp_path / "missing.ckpt"))


@needs_torch
def test_degrades_on_bad_input():
    from awpredict.core.concept import ConceptWorldModel
    m = ConceptWorldModel(d_in=D_IN)
    assert m.encode([[0.0] * 3]) is None          # wrong width
    assert m.surprise([[0.0] * D_IN]) is None     # shorter than two chunks
    assert m.plan([[0.0] * D_IN] * 8, []) == []


@needs_torch
def test_prediction_is_a_residual_on_the_current_concept():
    # Predicting the absolute next concept regressed to the mean and ranked
    # real-text drafts at chance; the delta form is what made plan() work.
    import torch
    from awpredict.core.concept import ConceptWorldModel
    m = ConceptWorldModel(d_in=D_IN, dim=64, k=K, segments=4, codes=16, levels=2,
                          layers=1, heads=4, max_concepts=32)
    for head in m.net.heads:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    for book in m.net.books:                     # uniform logits -> mix = mean code
        book.raw.data.zero_()
        for p in book.mlp.parameters():
            p.data.zero_()
    c = torch.randn(1, 5, 64)
    assert torch.allclose(m.net.predict(c), c, atol=1e-5)
