# Neighbour statistics of the benchmark systems

Cutoffs: 6.0 A (OMAT-0, MP-0, MPtrj, MH-1, POLAR-1) and 5.0 A (OFF23 / OFF24 / OMOL-0). Skin 0.5 A is not included: these are the edges inside the cutoff (the active edges), not the list's capacity.

## srtio3: 5000 atoms, species [('O', 3000), ('Sr', 1000), ('Ti', 1000)]

| cutoff | directed edges | edges / atom (mean, min, max) | 0.0-1.5 A | 1.5-2.0 A | 2.0-2.5 A | 2.5-3.0 A | 3.0-4.0 A | 4.0-5.0 A | 5.0-5.5 A | 5.5-6.0 A |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 249,922 | 50.0 (44, 52) | 0.0% | 3.5% | 1.3% | 19.2% | 17.3% | 58.7% | 0.0% | 0.0% |
| 6.0 | 369,266 | 73.9 (62, 86) | 0.0% | 2.4% | 0.8% | 13.0% | 11.7% | 39.7% | 6.1% | 26.3% |

Directed species pairs at 6.0 A (receiver, sender): O-O 34.2%, Ti-O 16.1%, O-Ti 16.1%, Sr-O 9.8%, O-Sr 9.8%, Sr-Sr 4.9%, Ti-Ti 4.9%, Sr-Ti 2.2%, Ti-Sr 2.2%

Same-species edges: 44.0%. Edges at d >= 2.75 A: 91.2%; d >= 5.0 A: 32.3%.

## nbcl5: 1944 atoms, species [('Cl', 1620), ('Nb', 324)]

| cutoff | directed edges | edges / atom (mean, min, max) | 0.0-1.5 A | 1.5-2.0 A | 2.0-2.5 A | 2.5-3.0 A | 3.0-4.0 A | 4.0-5.0 A | 5.0-5.5 A | 5.5-6.0 A |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 31,704 | 16.3 (14, 18) | 0.0% | 0.0% | 8.6% | 3.6% | 58.3% | 29.4% | 0.0% | 0.0% |
| 6.0 | 52,410 | 27.0 (22, 34) | 0.0% | 0.0% | 5.2% | 2.2% | 35.3% | 17.8% | 15.9% | 23.6% |

Directed species pairs at 6.0 A (receiver, sender): Cl-Cl 63.5%, Nb-Cl 17.6%, Cl-Nb 17.6%, Nb-Nb 1.4%

Same-species edges: 64.9%. Edges at d >= 2.75 A: 92.6%; d >= 5.0 A: 39.5%.

## nacl216: 216 atoms, species [('Cl', 108), ('Na', 108)]

| cutoff | directed edges | edges / atom (mean, min, max) | 0.0-1.5 A | 1.5-2.0 A | 2.0-2.5 A | 2.5-3.0 A | 3.0-4.0 A | 4.0-5.0 A | 5.0-5.5 A | 5.5-6.0 A |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 5,596 | 25.9 (25, 26) | 0.0% | 0.0% | 0.0% | 23.2% | 35.6% | 41.2% | 0.0% | 0.0% |
| 6.0 | 6,912 | 32.0 (32, 32) | 0.0% | 0.0% | 0.0% | 18.8% | 28.8% | 33.4% | 2.2% | 16.8% |

Directed species pairs at 6.0 A (receiver, sender): Na-Na 28.1%, Cl-Cl 28.1%, Na-Cl 21.9%, Cl-Na 21.9%

Same-species edges: 56.2%. Edges at d >= 2.75 A: 95.4%; d >= 5.0 A: 19.0%.

## water216: 648 atoms, species [('H', 432), ('O', 216)]

| cutoff | directed edges | edges / atom (mean, min, max) | 0.0-1.5 A | 1.5-2.0 A | 2.0-2.5 A | 2.5-3.0 A | 3.0-4.0 A | 4.0-5.0 A | 5.0-5.5 A | 5.5-6.0 A |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 34,778 | 53.7 (48, 61) | 2.9% | 1.5% | 3.7% | 7.3% | 33.6% | 51.0% | 0.0% | 0.0% |
| 6.0 | 54,848 | 84.6 (76, 95) | 1.8% | 0.9% | 2.4% | 4.6% | 21.3% | 32.3% | 19.8% | 16.8% |

Directed species pairs at 6.0 A (receiver, sender): H-H 45.8%, O-H 22.0%, H-O 22.0%, O-O 10.2%

Same-species edges: 56.0%. Edges at d >= 2.75 A: 92.9%; d >= 5.0 A: 36.6%.

## water: 5184 atoms, species [('H', 3456), ('O', 1728)]

| cutoff | directed edges | edges / atom (mean, min, max) | 0.0-1.5 A | 1.5-2.0 A | 2.0-2.5 A | 2.5-3.0 A | 3.0-4.0 A | 4.0-5.0 A | 5.0-5.5 A | 5.5-6.0 A |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 278,372 | 53.7 (46, 62) | 2.9% | 1.5% | 3.7% | 7.3% | 33.7% | 51.0% | 0.0% | 0.0% |
| 6.0 | 438,204 | 84.5 (75, 96) | 1.8% | 0.9% | 2.4% | 4.6% | 21.4% | 32.4% | 19.8% | 16.7% |

Directed species pairs at 6.0 A (receiver, sender): H-H 45.7%, O-H 22.0%, H-O 22.0%, O-O 10.3%

Same-species edges: 56.0%. Edges at d >= 2.75 A: 92.9%; d >= 5.0 A: 36.5%.

