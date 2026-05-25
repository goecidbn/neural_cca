Developer Guide
===============

Setting up a Development Environment
-------------------------------------

Clone the repository and install in editable mode with all development
dependencies::

    git clone https://github.com/goecidbn/neural_cca.git
    cd neural-cca
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -e ".[all]"

The ``[all]`` extra installs test, docs, linting, and batch-processing
dependencies in one step. You can also install subsets::

    pip install -e ".[test]"    # pytest + pytest-cov only
    pip install -e ".[docs]"    # Sphinx + PyData theme only
    pip install -e ".[dev]"     # ruff only

Running the Test Suite
----------------------

Run all tests with verbose output::

    pytest

Run with coverage report::

    pytest --cov=neural_cca --cov-report=term-missing

Run a single test class or test::

    pytest tests/test_tuning.py::TestOSI
    pytest tests/test_tuning.py::TestOSI::test_sharply_tuned_neuron_high_osi

Linting and Formatting
----------------------

The project uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and
formatting::

    ruff check .              # lint
    ruff check --fix .        # lint + auto-fix
    ruff format .             # format
    ruff format --check .     # check formatting without changing files

CI runs ``ruff check`` and ``ruff format --check`` on every push and PR.

Building the Documentation
--------------------------

Build and preview the Sphinx docs locally::

    cd docs
    make serve

This builds the HTML and starts a local server at ``http://localhost:8000``.
Press ``Ctrl+C`` to stop the server.

.. note::

   The documentation uses the `pydata-sphinx-theme`, which relies on Bootstrap
   JavaScript for layout and navigation. Opening the generated HTML files
   directly via ``file://`` in a browser will result in unstyled pages because
   browsers block JavaScript ``fetch()`` calls under the ``file://`` protocol.
   Always use ``make serve`` (or any local HTTP server) for previewing.

To build without serving (e.g. in CI)::

    cd docs
    make html

The output is written to ``docs/_build/html/``. On CI, docs are built and
deployed to GitHub Pages automatically on pushes to ``main``.

To clean and rebuild::

    cd docs
    make clean && make html


Easier sometimes::

    sphinx-build -b html -w warnings_sphinx_build.txt docs docs/_build/html
    python -m http.server  

This approach also generated a warnings log file that can be useful for debugging doc build issues.

Package Structure
-----------------

The package contains three submodules:

``sorting``
    Spike sorting (KMeans, PCA, quality metrics, batch processing).

``tuning``
    Orientation selectivity, tuning bandwidth, F0/F1/F2 modulation ratios.

``sta``
    Spike train statistics (ISI, CV, firing rates, refractory period violations).

Each submodule has its own ``utils.py`` with shared helpers (``guarded_divide``,
``steps2degree``). A canonical copy also lives in ``common_utils.py`` at the
package root.

Cross-submodule imports use relative imports (e.g.,
``from ..tuning.tuning import get_os_metrics``). External dependencies
on ``visioniceio`` are lazy-loaded inside ``try/except ImportError`` blocks so
the package works standalone.

Release Checklist
-----------------

1. Update the version in ``pyproject.toml`` (the ``[project].version`` field).
   ``neural_cca.__version__`` reads from there automatically.
2. Update ``docs/changelog.md``.
3. Run the full test suite: ``pytest``.
4. Build and check the distribution: ``python -m build && twine check dist/*``.
5. Tag the release: ``git tag v0.x.y && git push --tags``.
6. Upload to PyPI: ``twine upload dist/*``.
    6.1 test upload to TestPyPI first: ``twine upload --repository testpypi dist/*``.
