# learning_guards_signed_XOR

Code and outputs for

> J. Marco de Lucas, *From experience to reusable rules I: signed mismatch and learned applicability in a minimal world model* (2026). arXiv: (to be added).

Three small NumPy scripts reproduce every number in the note. They are deliberately minimal: a one-dimensional box world, a signed comparison, a fixed canonicalisation, a statistical evidence rule for a learned guard, and an exhaustive test.

## Contents

| file | what it does | tables / figures of the note |
|---|---|---|
| `code/cajas_pilot.py` | the binary box world; signed comparison; the available-extent code `H` and the width code `W`; the pooled and per-position mismatch units; the learned guard (Beta evidence rule) and its audit; the exhaustive 27,264-case test and the no-pooling ablation | Tables 2 and 3; Section 4–5 |
| `code/operator_pilot.py` | adds box identities (colours) to the accepted transitions; the canonicalisation staircase 875 → 81 → 6 → 2 → 1 with validation of the schema; reuse of the operator with withheld colours | Table 5; Section 7 |
| `code/circuit_pilot.py` | the same canonicalisation built from vector operations (signed bit comparator, salience OR, first-change anchor, propagation halted at the end of the run, coincidence check, sparse engram) and its ablations; writing the effect with the same routing | Section 7, "A concrete circuit check" |
| `code/paper_numbers.py` | five seeds, errors by width, size of the exhaustive set | Appendix A; per-width breakdown in Section 5 |
| `results/` | the printed output of each script and `paper_numbers.json` | — |

`operator_pilot.py` and `circuit_pilot.py` import `cajas_pilot.py` from the same directory; run everything from `code/`.

## Requirements

Python 3.10+ and NumPy (tested with Python 3.12 and NumPy 2.4). Nothing else.

```bash
cd code
python3 cajas_pilot.py      # ~1 min: guard, audit, exhaustive test, ablation
python3 operator_pilot.py   # staircase and reuse
python3 circuit_pilot.py    # circuit version and ablations
python3 paper_numbers.py    # five seeds (a few minutes)
```

All random seeds are fixed in the scripts; the outputs in `results/` are what the scripts print.

## What the code does and does not claim

The relation "free run ≥ box width" is never written as a rule or feature: it is available because both quantities are represented in a common thermometer code and compared with sign (Proposition 1 of the note). What is learned is which comparator signals inhibit the action, from accepted and refused attempts. The canonicalisation (select, relocate, substitute, bind) is supplied, not learned. See Section 10 of the note for the full statement of what is given, observed and learned.

## Companion note

The canonicalisation assumed here is built as a circuit, and tested in a second, relational world, in *From experience to reusable rules II: a minimal circuit for event-centred canonicalisation across relational spaces* (repository `canonicalisation_circuit_signed_XOR`, to be added).

## Licence

Code: MIT (see `LICENSE`). Text and figures of the note: see the arXiv listing.

## Citation

See `CITATION.cff`.
