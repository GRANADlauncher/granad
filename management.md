# Contributor guide

For a project overview and basic installation, start with [README.md](README.md).
The commands below are for a source checkout on Linux or a compatible Bash
environment. On Windows, use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
for these shell commands. Run them from the repository root unless stated otherwise.

## Environment and dependencies

Use Python 3.11 or newer in an isolated environment. Conda is optional; a venv is
sufficient. For example, create and activate a development environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Keep the environment directory out of commits. With Conda, activate your chosen
environment instead. Always use `python -m pip` so installation targets the same
interpreter as the checks.

[pyproject.toml](pyproject.toml) is the dependency source of truth. It declares
JAX, Diffrax, Matplotlib, NumPy, and SciPy as runtime dependencies. Geometry
construction also uses Matplotlib and SciPy, so they are required without
plotting. JAX selects its compatible JAXlib version. Accelerator setup follows
the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).

Install the extra needed for your work:

```bash
python -m pip install -e '.[test]'
# For documentation work instead:
python -m pip install -e '.[docs]'
```

Use `'.[test,docs]'` if you need both. The test extra provides pytest; the docs
extra provides MkDocs, pymdown extensions, mkdocstrings with its Python handler,
Jupytext, nbconvert, and ipykernel. Keep dependency declarations centralized
in pyproject.toml rather than repeating installation lists in workflows.

## Tests

The lightweight suite checks chain spectra, basis transformations, Rabi
oscillations, relaxation, simulation outputs, Fourier transforms, response,
periodic bands, XYZ round-trips, mean-field filling, and performance regressions.
It uses tiny systems and analytical or independent numerical references without
executing notebooks:

```bash
python -m pytest -q test/test_doc_invariants.py test/test_numerics.py test/test_mean_field.py test/test_performance.py
```

For subsystem work, also run the relevant existing tests, such as
`python -m pytest -q test/test_hubbard.py` for interacting mean-field behavior.
Run the full suite with:

```bash
python -m pytest
```

Report collection errors and failures separately from passing tests. A selected
passing subset does not establish a passing full suite. Headless plotting checks
can use `MPLBACKEND=Agg`; deliberately CPU-only runs can use `JAX_PLATFORMS=cpu`.

## Clean installation check

Use this after changing packaging or dependencies. An editable installation in
an existing environment cannot establish that all dependencies are declared.
Install normally in a fresh venv, check an import outside the checkout, then
install the test extra and exercise the installed package:

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

Success means the import path is inside this temporary environment's
`site-packages`, there are no broken requirements, and the numerical tests pass.
Repeat using Python 3.11 and the Python version configured in CI (currently 3.12).
Keep temporary artifacts until failures are diagnosed. A wheel build or install
dry-run alone does not verify a fresh installation or compatibility with newly
resolved dependency versions.

## Documentation

The documentation combines static pages in `docs/` with executable Python
tutorials in `docs_src/tutorials/` and the landing page in `docs_src/index.py`.
`docs/api.md` uses mkdocstrings directives to render API documentation from
`src/granad/`. Jupytext executes the tutorial sources as notebooks; nbconvert
turns them into Markdown and images; MkDocs renders the site using `mkdocs.yml`.

After installing the docs extra, register the active environment as the `base`
kernel expected by the tutorial metadata, then build:

```bash
python -m ipykernel install --sys-prefix --name base --display-name base
bash -e -o pipefail build_docs.sh
mkdocs build
```

To validate docs in the clean-install environment above, first run
`"$granad_python" -m pip install "${granad_repo}[docs]"` and
`source "$granad_check/venv/bin/activate"`. The kernel must use that environment's
interpreter. For a single tutorial, pass its filename to the script, for example
`bash -e -o pipefail build_docs.sh xyz.py`. Preview the built pages with
`mkdocs serve`.

The validation stages establish different things:

| Check | What it establishes |
| --- | --- |
| Lightweight numerical tests | Selected physical identities and API behavior |
| Full pytest suite | The broader set of collected test assertions |
| Tutorial execution and conversion | Documentation examples run and generate output |
| `mkdocs build` | Available static/API content renders into a site |
| Visual inspection | Figures, explanations, and layout look correct |

Use the shell error flags shown above: the build script can otherwise mask
execution or conversion failures. MkDocs alone can succeed without generated
tutorials, so it does not replace tutorial execution. The full docs build covers
larger examples, including mean field and topology, outside the lightweight suite.
Scripts under `docs_src/Time_Domain_Simulations/` are not included in this build.

The build generates notebooks, Markdown, images, and `site/`; tutorials may also
write other files. Review `git status` before and after, and keep generated output
out of source commits unless it is intentionally being maintained. Notebook
execution needs local kernel sockets. If the home directory is unwritable, put
Jupyter, IPython, Matplotlib, and cache directories in a writable temporary
directory. Report remaining environment restrictions separately from code errors.

## CI and publishing

The workflows in [.github/workflows/](.github/workflows) currently use Python 3.12
and opt in through the pushed commit's message:

| Flag | Branches | Action |
| --- | --- | --- |
| `[run-tests]` | `main` | Install the test extra and run pytest |
| `[build-docs]` | `main`, `dev` | Install the docs extra, execute tutorials, build and publish the site |

Neither workflow currently validates pull requests. A green workflow without its
flag may have skipped its substantive checks. `[build-docs]` also publishes; use
the local commands above when you only want validation.

`main` is the published branch, `dev` is the development branch, and `gh-pages`
contains generated documentation. Confirm the intended branch before switching
or merging. For a release, review and validate the changes being promoted to
`main`; account for the workflow triggers before pushing. Update documentation
sources rather than editing `gh-pages` directly.

| Branch | Published documentation | Example notebook |
| --- | --- | --- |
| `main` | [Stable site](https://granadlauncher.github.io/granad/) | [Linear response](https://github.com/GRANADlauncher/granad/blob/gh-pages/tutorials/i_linear_response.ipynb) |
| `dev` | [Development site](https://granadlauncher.github.io/granad/dev/) | [Linear response](https://github.com/GRANADlauncher/granad/blob/gh-pages/dev/tutorials/i_linear_response.ipynb) |

Main publishes at the site root; dev publishes under `/dev/`.

## External scripts

The [granad-scripts repository](https://github.com/granadlauncher/granad-scripts)
contains experimental features and utilities used in publications and projects
that are not part of GRANAD itself.
