# LoupMD manuscript draft

This draft contains the Methods and Related Work sections. [Read the PDF](main.pdf).

## Build

Install pdfLaTeX and BibTeX with the standard LaTeX packages, Latin Modern, microtype, and natbib. Run the manuscript build from the repository root.

```sh
make -C paper
```

The build writes `paper/build/main.pdf`. The four input files are `main.tex`, `methods.tex`, `related-work.tex`, and `references.bib`.

## Sources and evidence

- [Evidence and reproduction notes](evidence.md) link implementation claims to the code at the audited revision, identify the profiling record and document the POLAR validation inputs.
- [Citation map](citation_map.json) connects bibliography keys to locations in the manuscript and gives source passages for optimisation claims.
- [Editorial review](review.md) records the attribution and technical corrections made during drafting.
- The implementation was audited at [LoupMD revision `430c4aa`](https://github.com/FZWINGEL-UBT/LoupMD/tree/430c4aa4256c801b5a2a44dfdea713fcf31b4de7). Consult [the model table and checkpoint hashes](https://github.com/FZWINGEL-UBT/LoupMD/blob/430c4aa4256c801b5a2a44dfdea713fcf31b4de7/docs/reference/models.md) for model identities.

The three Materials Project CIFs used by the development POLAR validation are external inputs. Their identifiers, expected filenames and checksums are listed in the evidence notes. The [structure provenance record](https://github.com/FZWINGEL-UBT/LoupMD/blob/430c4aa4256c801b5a2a44dfdea713fcf31b4de7/loupmd/bench/structures/SOURCES.md) documents their redistribution status. Building the manuscript only requires the files in this directory.
