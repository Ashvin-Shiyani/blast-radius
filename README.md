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

## Data files

Every data file the image ships or builds is made from the pinned sqlglot commit (eb11aec) alone.
None stores anything per patch or per kind of edit, and none says which tests an edit breaks:
every decision about a commit is derived from that commit's diff when the selector runs.

- `map/blast.map` (shipped): for each test, the sqlglot lines it executed (function bodies and
  import-time code apart), the files it opened and the classes it instantiated. Made by running
  each test of the pinned checkout alone in a forked process, in this image's environment.
- `map/cmp.json` (shipped): for each `is`, `is not`, `==`, `!=` comparison in the sqlglot package,
  the pairs of operand classes that met there while the test suite ran; for each such class, which
  class defines its `__eq__` and `__ne__` and whether it is an Enum; the list of tests that ran and
  the content digest of the checkout. No test names per comparison, no verdicts. Made by running
  the suite with every such comparison wrapped in a recorder, four times in this image's
  environment (each test in its own forked process, and as one session; two hash seeds).
- `map/cost.json` (shipped): CPU seconds each test takes in the selector's own test runner. Made
  by `selector/costmap.py` on the pinned checkout, measured once on a development machine.
- `map/refs.json` (built with the image by `selector/refindex.py`): the file and line of every
  mention of each identifier in the checkout.
- `map/replay.json` (built with the image by `selector/replay.py build`): hashes of what sqlglot
  computes for each TPC query at the pinned commit, and the lines that run after optimisation.

When a patch is exactly one `is` <-> `==` or `is not` <-> `!=` swap, `selector/static.py` reads
the classes recorded at that comparison and works out from Python's comparison rules whether
equality and identity can differ for them. If they cannot, no test can observe the change and
none is selected; otherwise the usual rules apply. `BR_CMP=0` turns this off.

Outside the data files, the only numbers tied to a kind of change are the `GAIN_*` constants in
`selector/main.py`: how often running the candidates first pays off for an edit inside a function
body, for a declaration, and for a replayed TPC test. They only decide whether candidates are run
against the patched code before being selected.

Third-party code: sqlglot (MIT, fetched at build time) and the packages in
`requirements.lock` (the kit's environment). Python standard library otherwise.
