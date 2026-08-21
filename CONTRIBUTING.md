# Contributing

This is a small, research-stage package. Contributions are welcome —
especially another negative result, a bug in the engines, or a new
`EnvironmentAdapter` for a domain we haven't tried.

## Setup

```bash
pip install -e ".[torch]"
python -m pytest tests/ -v
```

## Ground rules

- **Degrade loudly, never silently.** An engine that can't operate (torch
  missing, checkpoint unreadable) sets `ok = False` and returns `None`/`[]`
  from its methods — it must never raise into a caller's turn loop, and it
  must never fabricate a prediction. This is the one rule every PR is held
  to; see `RESULTS.md` for why it mattered more than any architecture choice.
- **A trainer needs a positive assertion, not just a passing shape test.** A
  test that only checks a training method returns the right dict keys cannot
  tell a fitted model from a randomly-initialized one. If you add or change
  training code, assert it actually reduces loss on a target it can fit —
  see `tests/test_value_head.py` for the pattern.
- **New engine or adapter code should carry its own test file**, not rely on
  the existing ones happening to exercise it.

## Reporting a result

If you run this against your own environment and get a result — positive or
negative — open an issue or a PR adding it to `RESULTS.md`. Negative results
that survive someone trying to break them are the most useful thing this
repo can publish.
