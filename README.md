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
- `selector/pack.py` splits the selection over 1–8 workers to minimise the scored cost.

Third-party code: sqlglot (MIT, fetched at build time) and the packages in
`requirements.lock` (the kit's environment). Python standard library otherwise.
