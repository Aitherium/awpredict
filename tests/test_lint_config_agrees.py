"""The two ruff invocations that judge this package must not disagree.

This package is a directory inside a monorepo, and two different ruff runs look
at it:

  * the quality gate runs `ruff --isolated` from the REPO ROOT, where first-party
    detection cannot see this directory, so `awpredict` is third-party;
  * anything run from inside the package picks up this pyproject.toml and, by
    default, detects `awpredict/` as first-party.

Those two produce mutually exclusive I001 verdicts on the same import block, and
each `--fix` undoes the other. Measured: that really happened, twice, and it is
invisible from either side alone — each invocation is individually correct and
individually clean-able. `src = []` in pyproject.toml is what makes them agree.

The guard belongs here rather than in a comment because a comment did not stop it
the first time. Deleting that one line makes this test fail with the two verdicts
side by side, which is the only form of this problem anyone can act on.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent
HAVE_RUFF = shutil.which("ruff") is not None or bool(
    subprocess.run([sys.executable, "-m", "ruff", "--version"],
                   capture_output=True).returncode == 0
)


def _ruff(*args: str) -> tuple[int, str]:
    # encoding= is not optional: ruff's diagnostics contain em-dashes and box
    # characters, and text=True would decode them with the LOCALE codec (cp1252
    # here). The resulting UnicodeDecodeError is a ValueError, which no
    # OSError/SubprocessError guard catches — so the guard would crash instead of
    # reporting a verdict, on a machine where the lint is perfectly fine.
    r = subprocess.run([sys.executable, "-m", "ruff", "check", *args, str(PKG)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=PKG.parents[3])
    return r.returncode, (r.stdout + r.stderr)


# skipif at DECORATION time, never pytest.skip() in the body: a skip inside the
# body fires after partial execution and reports a real failure as "skipped".
@pytest.mark.skipif(not HAVE_RUFF, reason="ruff is not installed")
def test_isolated_and_config_rooted_ruff_give_the_same_verdict():
    isolated_rc, isolated_out = _ruff("--isolated", "--select", "E,F,I,N,W",
                                      "--ignore", "E501,E402,E741,N806,N812")
    ambient_rc, ambient_out = _ruff()
    assert isolated_rc == 0, (
        "the CI gate (`ruff --isolated` from the repo root) is unhappy:\n" + isolated_out)
    assert ambient_rc == 0, (
        "ruff using this package's own pyproject.toml is unhappy:\n" + ambient_out
        + "\n\nIf this passes isolated but fails here, the two configs have "
          "diverged again — check `src = []` under [tool.ruff].")
