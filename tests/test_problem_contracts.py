"""The Problem contract: a scorer that can refuse, records with one shape per role.

Every assertion here pins a rule the ARC-AGI-3 solving lane learned by losing time:
a scorer that returns nothing is not a scorer that scored zero; a NaN score is a bug,
not a data point; `kept` without a `baseline` is a claim without a comparison; and a
Fact may not leave the scope it was written in.
"""

import math

import pytest
from awpredict.contracts import (
    Attempt,
    Budget,
    EnvironmentAdapter,
    Fact,
    Outcome,
    ProblemSpec,
    Refusal,
    Score,
    Scorer,
    Transition,
    check_score,
    conforms,
    scope_problems,
)


# --------------------------------------------------------------------------- fakes
class _Env:
    domain = "toy"

    def __init__(self) -> None:
        self.state = 0

    def observe(self, env_state):
        return int(env_state)

    def actions(self):
        return [-1, 1]

    def step(self, action):
        self.state += action
        return self.state, float(self.state), self.state >= 3, {}


class _GoodScorer:
    metric = "reached"
    minimize = False

    def score(self, episode):
        if not episode:
            return Refusal("empty episode: nothing to judge")
        return Score(metric=self.metric, value=float(len(episode)))


class _BadScorer:
    """The defect shape: 'could not judge' expressed as nothing."""

    metric = "reached"
    minimize = False

    def score(self, episode):
        return None


# ------------------------------------------------------------------ conformance
def test_environment_adapter_conformance_names_what_is_missing():
    assert conforms(_Env(), EnvironmentAdapter) == []

    class Half:
        domain = "x"

        def observe(self, s):
            return s

    assert conforms(Half(), EnvironmentAdapter) == ["actions", "step"]


def test_scorer_conformance():
    assert conforms(_GoodScorer(), Scorer) == []
    assert isinstance(_GoodScorer(), Scorer)


# ------------------------------------------------------------------- the scorer
def test_scorer_may_refuse_and_a_refusal_carries_why():
    out = _GoodScorer().score([])
    assert isinstance(out, Refusal)
    assert "empty" in out.reason
    assert check_score(out) == []


def test_refusal_without_reason_is_rejected():
    with pytest.raises(ValueError):
        Refusal("")


def test_scorer_returning_none_is_named_as_a_defect_not_a_zero():
    problems = check_score(_BadScorer().score([1, 2]))
    assert problems and "None" in problems[0]


def test_bare_number_and_foreign_type_are_defects():
    assert check_score(0.0)
    assert check_score(3)
    assert check_score({"value": 1.0})
    assert check_score(True)  # bool is not a number here either


def test_score_value_must_be_finite():
    with pytest.raises(ValueError):
        Score(metric="m", value=math.nan)
    with pytest.raises(ValueError):
        Score(metric="m", value=math.inf)
    with pytest.raises(ValueError):
        Score(metric="", value=1.0)


def test_better_than_respects_direction_and_strict():
    assert Score("m", 2.0).better_than(1.0)
    assert not Score("m", 1.0).better_than(2.0)
    assert Score("ms", 1.0, minimize=True).better_than(2.0)
    assert not Score("ms", 2.0, minimize=True).better_than(1.0)
    # no baseline: the first honest score is an improvement over nothing
    assert Score("m", 0.0).better_than(None)
    # a WRONG candidate is never better, however fast
    assert not Score("ms", 0.001, minimize=True, strict_ok=False).better_than(100.0)


# ------------------------------------------------------------------- the spec
def _spec(**over):
    base = dict(
        problem_id="arc:ls20",
        domain="arc",
        adapter_ref="awgym.envs.games:ARCAGI3Env",
        scorer_ref="awgym.evals.score_awgym:LevelScorer",
    )
    base.update(over)
    return ProblemSpec(**base)


def test_wellformed_spec_has_no_problems_and_round_trips():
    spec = _spec()
    assert spec.problems() == []
    again = ProblemSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.budget == Budget(steps=80)


def test_spec_refuses_malformed_refs():
    assert any("adapter_ref" in p for p in _spec(adapter_ref="awgym.envs.games").problems())
    assert any("scorer_ref" in p for p in _spec(scorer_ref=":Only").problems())
    assert any("scorer_ref" in p for p in _spec(scorer_ref="mod: Cls").problems())


def test_spec_refuses_a_budget_that_never_stops():
    spec = _spec(budget=Budget(steps=0, seconds=0, tokens=0))
    assert any("binds on no axis" in p for p in spec.problems())
    assert Budget(steps=-1).problems()


def test_spec_refuses_a_non_awm_scope():
    assert scope_problems("platform:*:*") == []
    assert scope_problems("acme:alice:arc") == []
    assert scope_problems("platform") and scope_problems("a:b") and scope_problems("a::c")
    assert any("memory_scope" in p for p in _spec(memory_scope="just-arc").problems())


def test_planner_is_an_explicit_field_not_an_env_flag_chain():
    assert _spec().planner == "none"
    assert _spec(planner="mcts").planner == "mcts"


# ---------------------------------------------------------------- the records
def test_transition_round_trips_and_carries_the_domain():
    t = Transition(domain="toy", episode="e1", step=0, obs=0, action=1, next_obs=1,
                   reward=1.0, done=False, surprise=0.2)
    d = t.as_dict()
    assert d["domain"] == "toy" and d["surprise"] == 0.2
    assert Transition(**d) == t


def test_outcome_needs_exactly_one_verdict_and_a_baseline_before_kept():
    assert Outcome(problem_id="p", domain="d", episode="e").problems()
    both = Outcome(problem_id="p", domain="d", episode="e",
                   score=Score("m", 1.0), refusal=Refusal("x"))
    assert any("never both" in p for p in both.problems())
    unbased = Outcome(problem_id="p", domain="d", episode="e", score=Score("m", 1.0), kept=True)
    assert any("baseline" in p for p in unbased.problems())
    ok = Outcome(problem_id="p", domain="d", episode="e", score=Score("m", 1.0),
                 baseline=0.5, kept=True)
    assert ok.problems() == []
    refused = Outcome(problem_id="p", domain="d", episode="e", refusal=Refusal("no judge"))
    assert refused.problems() == []


def test_fact_is_scoped_and_bounded():
    good = Fact(memory_scope="acme:alice:arc", relation="win_condition", fact="reach the door")
    assert good.problems() == []
    assert Fact(memory_scope="arc", relation="r", fact="f").problems()
    assert Fact(memory_scope="a:b:c", relation="r", fact="f", confidence=1.5).problems()
    assert Fact(memory_scope="a:b:c", relation="", fact="f").problems()


def test_attempt_carries_score_or_refusal_and_serialises():
    a = Attempt(domain="toy", problem_id="p", episode="e", steps=3,
                score=Score("m", 2.0), opening_actions=[1, 1, 1], trajectory=[0, 1, 2])
    assert a.as_dict()["score"]["value"] == 2.0
    r = Attempt(domain="toy", problem_id="p", episode="e", steps=0, refusal=Refusal("empty"))
    assert r.as_dict()["refusal"]["reason"] == "empty"


# ------------------------------------------------------- one loop over the fakes
def test_a_toy_episode_produces_every_record_shape():
    env, scorer = _Env(), _GoodScorer()
    spec = _spec(problem_id="toy:1", domain="toy", adapter_ref="t:Env", scorer_ref="t:S")
    assert spec.problems() == []
    transitions = []
    obs, done, step = env.observe(env.state), False, 0
    while not done and step < spec.budget.steps:
        action = env.actions()[1]
        nxt, reward, done, _ = env.step(action)
        transitions.append(Transition(spec.domain, "e1", step, obs, action,
                                      env.observe(nxt), reward, done))
        obs, step = env.observe(nxt), step + 1
    verdict = scorer.score(transitions)
    assert check_score(verdict) == []
    out = Outcome(problem_id=spec.problem_id, domain=spec.domain, episode="e1",
                  score=verdict, baseline=0.0, kept=verdict.better_than(0.0),
                  transitions_n=len(transitions))
    assert out.problems() == [] and out.kept is True and out.transitions_n == 3
