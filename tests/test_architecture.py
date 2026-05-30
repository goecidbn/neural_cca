"""Architecture-invariant guards for the analysis leaf.

``neural_cca`` is one of two *leaf* packages in the Natal V1 working
tree (see ``../../CLAUDE.md`` → dependency direction).  The hard rule:

    visioniceio (I/O)  ← NEVER →  neural_cca (analysis)

Only the bridge (``vision_ice_analysis``) is allowed to compose the
two.  ``batch_sort_experiment`` used to violate this by importing
``visioniceio.experiment.Experiment`` directly; it now lives in the
bridge.  These tests fail loudly if that coupling ever creeps back.
"""

from __future__ import annotations

import ast
import pathlib

import neural_cca

_SRC_ROOT = pathlib.Path(neural_cca.__file__).resolve().parent


def _import_targets(tree: ast.AST):
    """Yield every dotted module name imported by *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or ""


def test_no_cross_leaf_import_of_visioniceio() -> None:
    """No module under ``neural_cca`` may import the I/O leaf."""
    offenders: list[str] = []
    for path in _SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, module in _import_targets(tree):
            if module == "visioniceio" or module.startswith("visioniceio."):
                offenders.append(f"{path.relative_to(_SRC_ROOT)}:{lineno}")
    assert not offenders, (
        "neural_cca (analysis leaf) must never import visioniceio "
        f"(I/O leaf); found: {offenders}. Route I/O composition through "
        "the vision_ice_analysis bridge instead."
    )


def test_importing_neural_cca_does_not_pull_in_visioniceio() -> None:
    """A fresh ``import neural_cca`` must not load ``visioniceio``.

    Run in a subprocess so an already-imported sibling in this test
    session cannot mask a real top-level import edge.
    """
    import subprocess
    import sys

    code = "import neural_cca, sys; assert 'visioniceio' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_batch_sort_experiment_is_gone_from_public_api() -> None:
    """The directory-loading batch driver moved to the bridge."""
    assert not hasattr(neural_cca, "batch_sort_experiment")
    from neural_cca import sorting

    assert "batch_sort_experiment" not in sorting.__all__
