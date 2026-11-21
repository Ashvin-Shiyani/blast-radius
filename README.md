# Blast Radius test selector

For each patch to sqlglot (pinned at eb11aec), pick the tests that can catch it and split
them across workers.

- `map/blast.map` is a coverage map built from the pinned commit only: every sqlglot test
  was run in isolation (one forked process per test, including the pool workers it
  spawns) and the lines it executed, split into function-body and import-time lines, and
  the data files it opened were recorded. No patch, answer or kind of edit was used to
  build it.
- `selector/static.py` finds the tests whose recorded execution touches the code a patch
  changes (statement level inside functions, importers for module/class-level code,
  openers for data files, everything for pytest config or unreadable patches).
- Class-body and module-level changes are narrowed to the tests that instantiate the class
  (recorded per test, subclasses included) or read the changed names (an identifier index of
  the pinned checkout, built at image build time by `selector/refindex.py`).
- When cheaper than running them all, the cheap candidates are run against the patched code
  (`selector/warm.py`: a forked pytest session per commit over a symlink overlay in /tmp) and
  only the ones that fail are kept; anything uncertain is kept.
- For a one-expression change inside a function, `selector/probe.py` rewrites the expression to
  evaluate both the original and the patched form while the cheap tests run; if the tests reach
  it and the two never differ, the change is invisible and the costly candidates are dropped.
- `selector/pack.py` splits the selection over 1–8 workers to minimise the scored cost.

Third-party code: sqlglot (MIT, fetched at build time) and the packages in
`requirements.lock` (the kit's environment). Python standard library otherwise.
