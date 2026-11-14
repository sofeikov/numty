# Bug-hunt benchmark

Real NumPy code from other projects, annotated with named sizes, plus
realistic seeded shape bugs. It comes from the bug hunt in
[#25](https://github.com/sofeikov/numty/issues/25). Use it to score a
change to the checker: more bugs caught, and no new errors on the clean
files.

| Target | Areas | Results |
|---|---|---|
| filterpy 2.0.0 | `kalman`, `ukf`, `ekf_ckf`, `info_sqrt` | [#338](https://github.com/sofeikov/numty/issues/338) |
| scipy.signal 1.18.1 | `lti_conversion`, `ltisys`, `savgol_zi` | [#339](https://github.com/sofeikov/numty/issues/339) |
| statsmodels 0.15.0 | `mnlogit`, `var`, `linear_model`, `pca` | [#340](https://github.com/sofeikov/numty/issues/340) |

Each area has these files:

- `clean.py`: a standalone extract that uses only numpy and numty. It is
  annotated, and should check clean.
- `harness.py`: deterministic calls with sizes that never coincide.
- `ref_check.py`: compares `clean.py` with the upstream library. It needs
  that library installed.
- `mutants/`: copies of `clean.py`, each with one seeded bug, described on
  its first line.
- `NOTES.md`: what was extracted, the `# NUMTY:` escapes, and the table of
  seeded bugs.

## Scoring

```sh
uv run python bench/hunt/score.py bench/hunt/filterpy/kalman            # static and runtime
uv run python bench/hunt/score.py --static-only bench/hunt/statsmodels/*  # static only
```

The scorer runs mypy in both modes: default, and `mypy-names.ini`. A
mutant counts as caught in a mode when it has an error that the clean
file lacks. At runtime, each mutant is classed as one of:

- `crash`: it raises.
- `silent`: it runs, but its outputs differ from the clean file's.
- `same`: its outputs equal the clean file's. It is not counted as a bug.

The per-area result goes to `score.json`, which is not tracked. Recorded
totals are in `_results/`, one file per numty commit.
