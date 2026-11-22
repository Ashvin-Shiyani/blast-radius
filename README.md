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

- `map/blast.map` (shipped): per test, the sqlglot lines it executed, the files it opened and the
  classes it instantiated. Made by running each test alone in a forked process, in this image's
  environment.
- `map/cmp.json` (shipped): per `is`, `is not`, `==`, `!=` comparison in the sqlglot package, the
  pairs of operand classes that met there while the test suite ran; per class, which class defines
  its `__eq__` and `__ne__` and whether it is an Enum; the tests that ran; the checkout's content
  digest. No test names per comparison, no verdicts. Made by running the suite with each such
  comparison wrapped in a recorder, in this image's environment (every test forked, and as one
  session; two hash seeds). When a patch is exactly one `is` <-> `==` or `is not` <-> `!=` swap,
  `selector/static.py` works out from these classes and Python's comparison rules whether the two
  operators can differ there; if not, no test can observe the change and none is selected.
  `BR_CMP=0` turns this off.
- `map/tab.json` (shipped): per dict table written out in a class body of the sqlglot package
  (`NAME = {...}`) and per entry of it, the tests that looked that entry's key up or read all
  values of the table while the test suite ran; the tests that ran; the checkout's content
  digest. An entry is left out if its key or the whole table was read while a module was being
  executed or outside a test (something derived from it may be kept elsewhere), and a table is
  left out if it was changed after it was built or was not built the same way in every run. No
  verdicts, nothing about any edit. Made by running the suite with each such table replaced by a
  dict that behaves the same and notes key lookups and reads of all values, in this image's
  environment (every test forked, and as one session; two hash seeds each). When a patch changes
  exactly one literal value of one listed entry to another literal, on one line,
  `selector/static.py` takes the tests listed for that entry as the candidates; for anything
  else its other rules decide. `BR_TAB=0` turns this off.
- `map/cost.json` (shipped): CPU seconds each test takes in the selector's own runner. Made by
  `selector/costmap.py` on the pinned checkout, measured once on a development machine.
- `map/refs.json` (built with the image by `selector/refindex.py`): where each identifier is mentioned.
- `map/replay.json` (built with the image by `selector/replay.py build`): hashes of what sqlglot
  computes for each TPC query at the pinned commit, and the lines that run after optimisation.

Outside the data files, the only numbers tied to a kind of change are the `GAIN_*` constants in
`selector/main.py`: how often running the candidates first pays off for a body edit, a declaration
and a replayed TPC test. They only decide whether candidates are run before being selected.

Third-party code: sqlglot (MIT, fetched at build time) and the packages in
`requirements.lock` (the kit's environment). Python standard library otherwise.
