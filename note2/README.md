# Note II — event-centred canonicalisation by path integration

Code and outputs for

> J. Marco de Lucas, *From experience to reusable rules II: event-centred canonicalisation by path integration across relational spaces* (2026). Zenodo (DOI to be added).

Companion to note I ([doi:10.5281/zenodo.23193217](https://doi.org/10.5281/zenodo.23193217)), whose code is in the root folders `code/` and `results/` of this repository. The filler codebook (six colours, six active bits of 64) is the same as in `circuit_pilot.py` of note I.

## Contents

| file | what it does | tables of the note |
|---|---|---|
| `code/canon_spaces_pilot.py` | four relational spaces (line, ring, grid, binary tree); the circuit: signed mismatch, onset anchor, path integration with one shift operator per generator, coincidence for the role X, signed comparison of extents; staircase; ablations; reuse with withheld colours, withheld shapes and larger environments; choice among four candidate frames; local codebooks, transport, a curved link, Wilson loops | Tables 3–5, 7 and 10; Sections 5, 7 and Appendix B |
| `code/heteroclinic_coordinate_pilot.py` | the path-integration primitive as an intrinsic heteroclinic sequence of coordinate populations (GLV) that the event resets and stops, on the line and ring: recognition compared with the breadth-first construction, execution and guard, a whole-ring event (integer chain, inhibition of return, heteroclinic cycle of known size L), dwell times, noise scan (100 realisations × 6 extents × 7 levels), stop-strength threshold | Table 9, Figure 3; Section 11 |
| `code/relations_pilot.py` | a hierarchy (organisation chart): the relation read by three propagators in competition (staircase and ablations); the guard of note I learned with the same Beta evidence rule (pooled vs per-position comparator, unseen persons and length); a branching relation and the relative address; on the line world, the correspondence between independent hand and shelf codes learned by one-shot heteroassociation (Hebbian, outgoing-weight normalisation, and renormalisation after every update as a control; 20 codebooks) | Table 6; Sections 6 and 9 |
| `results/salida_canon_spaces_pilot.txt` | printed output of the main script | — |
| `results/salida_heteroclinic_coordinate_pilot.txt`, `results/numbers_heteroclinic.json`, `results/fig_noise_scan.pdf` | output, numbers and noise-scan figure of the heteroclinic script | — |
| `results/salida_relations_pilot.txt`, `results/numbers_relations.json` | output and numbers of the relations script | — |
| `results/numbers_note2.json` | all numbers of the note | — |

## Requirements

Python 3.10+ and NumPy (tested with Python 3.12 and NumPy 2.4); SciPy for the evidence rule in `relations_pilot.py`.

```bash
cd note2/code
python3 canon_spaces_pilot.py              # under a minute
python3 heteroclinic_coordinate_pilot.py   # about 1.5 minutes; imports the main script
python3 relations_pilot.py                 # about ten seconds; imports the main script
```

Seeds are fixed (3: episodes; 5: codebook; 17: reuse; 23, 29: content frames; 41: noise on training placements and 1000+1000η: noise scan, in the heteroclinic script; 11, 13, 31, 37, 41, 43 and 100–119 in the relations script). The output in `results/` is what the script prints.

## What the code does and does not claim

Canonicalisation is written as a construction of local vector operations (a circuit interpretation is proposed, not tested): the anchor is the onset of the change (the positive channel of the signed comparison applied between neighbouring slots), and relocation is path integration of a bump reset at that anchor. The generators of each space, their shift operators, the onset rule and its priority, the shapes written in relative coordinates, the four candidate frames and the link maps are **supplied**. What is measured is what each of these requirements buys, in spaces where the shelf of note I could not tell them apart. Nothing here learns the structure of a space, and propagation is serialised by a breadth-first scheduler. On the line and ring, `heteroclinic_coordinate_pilot.py` replaces the scheduler by an intrinsic heteroclinic sequence that the event resets and stops; the routing from coordinate populations to slots is still supplied, and the GLV network is a normal form, not a connectome. See Sections 11–12 of the note.

## Licence

Code: MIT (see `LICENSE` at the repository root). Text and figures of the note: see the Zenodo record.
