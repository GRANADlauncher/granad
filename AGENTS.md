# Agent instructions

## Working rules

- Preserve unrelated staged and unstaged changes. Commit, push, or deploy only
  when requested. Do not add publishing flags to obtain local validation or edit
  the generated `gh-pages` branch.
- Keep changes small and follow existing patterns. Prefer analytical solutions
  and physical identities over saved numerical snapshots for regression tests.
- Use `pyproject.toml` as the dependency source of truth. Add version bounds only
  for demonstrated compatibility requirements; keep accelerator dependencies out
  of the default installation.

## Repository map

- `src/granad/`: runtime code.
- `test/`: numerical tests and exploratory scripts.
- `docs_src/tutorials/` and `docs_src/index.py`: executable documentation sources.
- `docs/`: static pages, API documentation, and generated tutorial output.
- [README.md](README.md): human entry point.
- [management.md](management.md): canonical setup, validation, and publishing
  commands. Link to this guide rather than duplicating its procedures here.

## Choosing validation

- For numerical changes, run the [lightweight tests](management.md#tests) and
  relevant subsystem tests; mean-field changes also need the Hubbard test.
- For packaging changes, follow the [clean installation check](management.md#clean-installation-check).
  An editable install or a wheel build alone does not verify a fresh install.
- For documentation changes, use the [documentation guide](management.md#documentation).
  Static rendering does not establish successful tutorial execution or correct
  plot appearance.
- Report the exact scope of passing checks and any failures. Distinguish
  environment restrictions from repository failures; a selected passing subset
  does not establish a passing full suite.
- Review generated artifacts before handing off changes. Do not include them
  in source changes unless requested.

## Workstation environment

The existing `granad` Conda environment can be reused for local checks. If Conda
is unavailable in a noninteractive shell, initialize it with
`source ~/miniconda3/etc/profile.d/conda.sh` before `conda activate granad`.
Fresh-install verification still needs an isolated environment.
