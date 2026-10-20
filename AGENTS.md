# Working on GRANAD

## Scope and repository layout

- Preserve unrelated staged and unstaged changes. Commit, push, or deploy only
  when requested; local validation does not require publishing documentation.
- Runtime code is in `src/granad/`; tests are in `test/`.
- Executable tutorial sources are in `docs_src/tutorials/`, with the landing
  page in `docs_src/index.py`. Static pages and API documentation are in `docs/`.
- Keep changes small. Prefer analytical solutions and physical identities over
  saved numerical snapshots for regression tests.

## Environment and dependencies

Use Python 3.11 or newer in an isolated venv or Conda environment. On the current
workstation, an existing environment is available with `conda activate granad`.
If Conda is unavailable in a noninteractive shell, initialize it first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate granad
```

`pyproject.toml` is the source of truth for dependencies:

- Runtime: JAX, Diffrax, Matplotlib, NumPy, and SciPy. Matplotlib and SciPy are
  also used by geometry construction, so they are required without plotting.
- `test` extra: pytest.
- `docs` extra: MkDocs, pymdown extensions, mkdocstrings with its Python handler,
  Jupytext, nbconvert, and ipykernel.
- Let JAX select its compatible JAXlib. Add version bounds only for a demonstrated
  compatibility requirement; retain the setuptools build-system minimum.

Install only the extras needed for the task, using the active interpreter:

```bash
python -m pip install -e '.[test]'
# For documentation work instead:
python -m pip install -e '.[docs]'
```

Conda is optional. Accelerator setup follows the
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html);
do not add GPU-specific dependencies to the default package installation.

## Numerical validation

Run from the repository root. The lightweight checks exercise tutorial physics
and API consistency without executing notebooks:

```bash
python -m pytest -q test/test_doc_invariants.py test/test_numerics.py test/test_mean_field.py test/test_performance.py
```

Use `python -m pytest` for the full suite, and add the relevant existing tests
when changing a subsystem, for example `test/test_hubbard.py` for mean field.
Run plotting checks with `MPLBACKEND=Agg` in a headless environment. For deliberate
CPU-only validation, set `JAX_PLATFORMS=cpu`. Report collection errors and failures
explicitly; a selected passing subset does not establish a passing full suite.

## Clean installation check

Use this after changing packaging or dependencies. An editable installation in
the existing Conda environment cannot establish that dependencies are complete.
From the repository root, install normally in a fresh venv, check an import
outside the checkout, and then install the test extra:

```bash
granad_repo="$PWD"
granad_check="$(mktemp -d)"
python3 -m venv "$granad_check/venv"
granad_python="$granad_check/venv/bin/python"
"$granad_python" -m pip install --upgrade pip
"$granad_python" -m pip install "$granad_repo"
"$granad_python" -m pip check
(cd "$granad_check" && "$granad_python" -I -c 'import granad; print(granad.__file__)')
"$granad_python" -m pip install "${granad_repo}[test]"
"$granad_python" -I -m pytest -q test/test_doc_invariants.py test/test_numerics.py test/test_mean_field.py test/test_performance.py
```

Require the import path to be inside the temporary environment's `site-packages`,
no broken requirements, and passing numerical tests. Repeat using Python 3.11 and
the Python version configured in CI (currently 3.12). Keep temporary artifacts
until failures are diagnosed. A wheel build or install dry-run alone does not
verify a fresh installation or compatibility with newly resolved versions.

## Documentation validation

Install the `docs` extra in the environment being checked, then activate it.
For the clean-install environment above, use
`"$granad_python" -m pip install "${granad_repo}[docs]"` and
`source "$granad_check/venv/bin/activate"`. From the repository root:

```bash
python -m ipykernel install --sys-prefix --name base --display-name base
bash -e -o pipefail build_docs.sh
mkdocs build
```

The `base` kernel must use this environment's interpreter. A targeted tutorial
build accepts a filename, e.g. `bash -e -o pipefail build_docs.sh xyz.py`.
The shell error flags matter: the script itself can otherwise mask failed
execution or conversion. `mkdocs build` alone checks static/API rendering, not
tutorial execution. Plot appearance still requires visual inspection.

The build generates notebooks, Markdown, images, and `site/`, and tutorials may
write other files. Review `git status` before and after; do not include generated
artifacts in source changes unless requested. Notebook execution needs local
kernel sockets. In a sandbox with an unwritable home directory, put Jupyter,
IPython, Matplotlib, and cache directories under a writable temporary directory.
Report remaining environment restrictions separately from repository failures.

## CI and publishing

Read `.github/workflows/` before relying on CI status. Currently both workflows
use Python 3.12 and opt in through the pushed commit's message:

- `[run-tests]`: pytest on pushes to `main`.
- `[build-docs]`: execute tutorials, build the site, and publish on pushes to
  `main` or `dev`. Main publishes at the site root; dev publishes under `/dev/`.

Neither workflow currently validates pull requests. A green workflow without its
flag may have skipped all substantive checks. Do not add publishing flags merely
to validate a local installation, or edit the generated `gh-pages` branch.
