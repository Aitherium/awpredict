"""RoutingEnv satisfies EnvironmentAdapter, reads AXRL, and scores warm sets without peeking."""

import struct

import pytest
from awpredict.adapters.routing_env import (
    RoutingEnv,
    RoutingStep,
    WarmSetAction,
    multi_hot,
    read_axrl,
)
from awpredict.contracts import EnvironmentAdapter, conforms

L, E, K = 2, 8, 2


def _trace():
    # layer 0 alternates {0,1} / {2,3}; layer 1 always {4,5}
    return [
        RoutingStep(i, 0, {0: frozenset({0, 1} if i % 2 == 0 else {2, 3}), 1: frozenset({4, 5})})
        for i in range(6)
    ]


def _write_axrl(path, steps, tag=0):
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHHHHQ12x", b"AXRL", 1, L, E, K, 0))
        for st in steps:
            for layer, ids in sorted(st.layers.items()):
                ids = sorted(ids)
                f.write(struct.pack("<HHHHHHII", 0xA17E, layer, 1, len(ids), tag, 0, st.ubatch, 0))
                f.write(struct.pack(f"<{len(ids)}H", *ids))
        f.write(b"\x7e\xa1\x00")  # writer mid-record: dropped, not an error


def test_conforms_to_the_environment_adapter_protocol():
    env = RoutingEnv(_trace(), L, E, keep=2)
    assert conforms(env, EnvironmentAdapter) == []
    assert isinstance(env, EnvironmentAdapter)
    assert env.domain == "moe_routing"


def test_observe_actions_step_score_the_next_ubatch():
    env = RoutingEnv(_trace(), L, E, keep=2)
    assert env.observe() == (frozenset({0, 1}), frozenset({4, 5}))
    acts = {a.name: a for a in env.actions()}
    assert set(acts) == {"last_ubatch", "frequency", "blend"}
    assert acts["last_ubatch"].sets == (frozenset({0, 1}), frozenset({4, 5}))
    # next ubatch selects {2,3} at layer 0: LRU keeps {0,1} -> half the experts hit
    _, reward, done, info = env.step(acts["last_ubatch"])
    assert (info["hits"], info["needed"], reward, done) == (2, 4, 0.5, False)
    assert env.observe() == (frozenset({2, 3}), frozenset({4, 5}))
    perfect = WarmSetAction("oracle-for-test", (frozenset({0, 1}), frozenset({4, 5})))
    assert env.step(perfect)[1] == 1.0
    assert len(multi_hot(env.observe(), E)) == L * E


def test_episode_ends_and_rejects_wrong_layer_count():
    env = RoutingEnv(_trace(), L, E, keep=4)
    with pytest.raises(ValueError):
        env.step([frozenset()])
    done = False
    n = 0
    while not done:
        _, _, done, _ = env.step(env.actions()[1])
        n += 1
    assert n == 5
    assert env.step(env.actions()[0])[2] is True


def test_read_axrl_roundtrip_with_rotation(tmp_path):
    p = tmp_path / "route.axrl"
    tr = _trace()
    _write_axrl(str(p) + ".1", tr[:4], tag=3)
    _write_axrl(p, tr[4:])
    h, steps = read_axrl(p)
    assert (h.n_layer, h.n_expert, h.n_expert_used) == (L, E, K)
    assert [s.layers for s in steps] == [s.layers for s in tr]
    assert [s.stream for s in steps] == [0, 0, 0, 0, 1, 1] and steps[0].tag == 3
    env = RoutingEnv.from_axrl(p, keep_fraction=0.25)
    assert env.keep == 2 and len(env.steps) == 6


def test_empty_log_is_an_error_not_an_empty_env(tmp_path):
    p = tmp_path / "route.axrl"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        read_axrl(p)
