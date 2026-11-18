# Models

LoupMD runs the released MACE checkpoints of a fixed label table (`loupmd/models.py`). A label names one checkpoint
file and, for a multi-head checkpoint, one head. `loupmd models` prints the table; `loupmd models --long` adds the
full SHA-256, the URL and the source of each training functional. How to choose and load a model is in
[Models](../guide/models.md).

## Label table

The table below mirrors `LABELS` in `loupmd/models.py`, in its order. "Checkpoint" is the checkpoint's name in
`CHECKPOINTS`. "Head" is the head the label selects; `-` marks a checkpoint without stored heads. "Functional" is the
training functional of that head, which selects the D3 parameter set of a dispersion correction
([Dispersion](../guide/dispersion.md)); "unknown" means the release names none. SHA-256 shows the first 12 hex digits
of the released file's hash, as `loupmd models` prints them. Cutoff and layers were read from the released files.

| Label | Checkpoint | File | Family | Model type | Head | Functional | Licence | SHA-256 (12) | Cutoff (A) | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mp0-medium` | `medium` | `2023-12-03-mace-128-L1_epoch-199.model` | residual | MACE | - | pbe | MIT | `01bfe2210013` | 6.0 | 2 |
| `mp0-large` | `large` | `MACE_MPtrj_2022.9.model` | residual | MACE | - | pbe | MIT | `f80e992b65ab` | 4.5 | 2 |
| `mp0b3-medium` | `medium-0b3` | `mace-mp-0b3-medium.model` | density | MACE | default | pbe | MIT | `2f2be696351a` | 6.0 | 2 |
| `mpa0-medium` | `medium-mpa-0` | `mace-mpa-0-medium.model` | density | MACE | default | pbe | MIT | `75428afe3a1d` | 6.0 | 2 |
| `omat0-medium` | `medium-omat-0` | `mace-omat-0-medium.model` | density | MACE | default | pbe | ASL | `d4b14be9afa2` | 6.0 | 2 |
| `omat0-small` | `small-omat-0` | `mace-omat-0-small.model` | density | MACE | omat_pbe | pbe | ASL | `0abfde07862c` | 6.0 | 2 |
| `matpes-pbe` | `mace-matpes-pbe-0` | `MACE-matpes-pbe-omat-ft.model` | density | MACE | default | pbe | ASL | `e618ad582b84` | 6.0 | 2 |
| `matpes-r2scan` | `mace-matpes-r2scan-0` | `MACE-matpes-r2scan-omat-ft.model` | density | MACE | default | r2scan | ASL | `8f147ecffa1d` | 6.0 | 2 |
| `mh1-omat` | `mh-1` | `mace-mh-1.model` | mh1 | MACE | omat_pbe | pbe | ASL | `a522eb7f59c7` | 6.0 | 2 |
| `mh1-mp` | `mh-1` | `mace-mh-1.model` | mh1 | MACE | mp_pbe_refit_add | pbe | ASL | `a522eb7f59c7` | 6.0 | 2 |
| `mh1-r2scan` | `mh-1` | `mace-mh-1.model` | mh1 | MACE | matpes_r2scan | r2scan | ASL | `a522eb7f59c7` | 6.0 | 2 |
| `polar1-s` | `polar-1-s` | `MACE-POLAR-1-S.model` | polar | PolarMACE | Default | unknown | ASL | `e4495612037b` | 6.0 | 1 |
| `polar1-m` | `polar-1-m` | `MACE-POLAR-1-M.model` | polar | PolarMACE | Default | unknown | ASL | `fab8b8713c83` | 6.0 | 2 |
| `polar1-l` | `polar-1-l` | `MACE-POLAR-1-L.model` | polar | PolarMACE | Default | unknown | ASL | `9f65f8dc6dda` | 6.0 | 3 |
| `omol0-xl` | `omol-0-xl` | `MACE-omol-0-extra-large-1024.model` | omol | MACE | omol | wb97m-v | ASL | `9b64b4fd5153` | 6.0 | 3 |
| `off23-small` | `off23-small` | `MACE-OFF23_small.model` | off | MACE | - | wb97m-d3bj | ASL | `165cce4cfec5` | 4.5 | 2 |
| `off23-medium` | `off23-medium` | `MACE-OFF23_medium.model` | off | MACE | - | wb97m-d3bj | ASL | `4842c52ad210` | 5.0 | 2 |
| `off23-large` | `off23-large` | `MACE-OFF23_large.model` | off | MACE | - | wb97m-d3bj | ASL | `a29e397dbf3e` | 5.0 | 2 |
| `off23b-medium` | `off23b-medium` | `MACE-OFF23b_medium.model` | off | MACE | - | wb97m-d3bj | ASL | `871653738a4f` | 6.0 | 2 |
| `off24-medium` | `off24-medium` | `MACE-OFF24_medium.model` | off | MACE | - | wb97m-d3bj | ASL | `e5ccf5837f68` | 6.0 | 2 |

Licences (`LICENCE_TEXT`):

| Code | Licence |
| --- | --- |
| ASL | Academic Software License (non-commercial use only), https://github.com/gabor1/ASL |
| MIT | MIT License |

LoupMD itself is MIT-licensed. The run report names the model's licence.

### Elements

Each checkpoint has an embedding for a fixed set of atomic numbers. A structure with another element is refused
with `ModelElementError` before anything is exported ([Errors](errors.md)).

| Checkpoints | Atomic numbers |
| --- | --- |
| Materials Project models (`mp0-*`, `mp0b3-medium`, `mpa0-medium`, `omat0-*`, `matpes-*`) and MH-1 (`mh1-*`) | 1-83 and 89-94 |
| POLAR-1 (`polar1-*`) and OMOL-0 (`omol0-xl`) | 1-83 |
| MACE-OFF (`off23-*`, `off23b-medium`, `off24-medium`) | 1, 6, 7, 8, 9, 15, 16, 17, 35, 53 (H, C, N, O, F, P, S, Cl, Br, I) |

### Other heads and aliases

MH-1 stores six heads, in this order: `matpes_r2scan`, `mp_pbe_refit_add`, `spice_wB97M`, `oc20_usemppbe`, `omol`,
`omat_pbe`. Three have labels. The others are reached through the checkpoint name or its alias and `head=`, for
example `lm.load("mh1", head="spice_wB97M")` or `--model mh1 --head spice_wB97M`. Such a head gets the label
`mh-1:<head>`, which `find` also accepts. A multi-head checkpoint without `head=` is refused with its heads listed,
also when its file is given by path; there is no default head.

| Alias | Checkpoint |
| --- | --- |
| `mh1` | `mh-1` |
| `omol0` | `omol-0-xl` |

| MH-1 head | Functional |
| --- | --- |
| `matpes_r2scan` | r2scan |
| `mp_pbe_refit_add` | pbe |
| `spice_wB97M` | wb97m-d3bj |
| `oc20_usemppbe` | pbe |
| `omol` | wb97m-v |
| `omat_pbe` | pbe |

`omol0-xl` has a total-charge and a total-spin embedding, one value per structure; the checkpoint's embeddings
cover total charge -100 to 100 and total spin 0 to 100. Set them with `--charge` and `--spin-multiplicity` (2S + 1),
or `charge=` and `spin_multiplicity=`; the defaults are 0 and 1. The export folds both into the bundle, so each
setting is its own bundle. Other models refuse them.

## Finding and trusting the file

A checkpoint is a pickle and runs code when it is loaded. LoupMD unpickles a file only when one of these holds:

- its SHA-256 matches the label table (a label);
- its SHA-256 matches the hash the caller gives (`sha256=`, `--sha256`);
- the caller says `trusted=True` (`--trusted`).

Otherwise `CheckpointTrustError` is raised. A URL without a hash is refused before it is downloaded.

A label's file is looked for in this order:

1. the explicit path (`checkpoint=`, `--checkpoint`);
2. `$LOUPMD_MODELS/<file>`;
3. `<cache root>/models/<file>`;
4. with `download=True` (`--download`, `loupmd models --download LABEL`) only: the official release URL, downloaded
   into `<cache root>/models/`. A download whose SHA-256 differs from the table is deleted before anything reads it.

An empty file is skipped. `LOUPMD_MODELS` and the cache root are described in [Environment](environment.md).

## Families and kernels

Every checkpoint belongs to one of the families in `FAMILIES`: `mh1`, `omol`, `polar`, `residual`, `density`, `off`.
The family selects the benchmark system of `loupmd bench --suite families` and, for `polar`, a refusal of
dispersion corrections (POLAR-1 models long-range interactions of their own).

Every model runs the generic kernels by default (`src/generic.cu`, `src/generated_generic.cu`, generated by
`codegen/gen_generic.py`). The export writes a generic bundle for ScaleShiftMACE models with RealAgnostic,
RealAgnosticResidual, RealAgnosticDensity, RealAgnosticDensityResidual and RealAgnosticResidualNonLinear interaction
blocks.

| Family | Labels | Kernels |
| --- | --- | --- |
| residual | `mp0-medium`, `mp0-large` | generic |
| density | `mp0b3-medium`, `mpa0-medium`, `omat0-medium`, `omat0-small`, `matpes-pbe`, `matpes-r2scan` | generic; `omat0-medium` also has specialised kernels (below) |
| mh1 | `mh1-omat`, `mh1-mp`, `mh1-r2scan` | generic |
| polar | `polar1-s`, `polar1-m`, `polar1-l` | generic for the local part, plus `src/polar.cu` for the long-range k-space electrostatics and the local-environment MLP |
| omol | `omol0-xl` | generic, with the charge and spin embedding folded into the bundle |
| off | `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium` | generic |

MACE-OMAT-0 medium (`omat0-medium`) also has kernels specialised to its architecture (`src/engine.cu`,
`src/interaction0.cu`, generated by `codegen/gen_kernels.py`). They are about 8% faster on the RTX 5060 Ti and not
faster on the A100, so they are not the default. `generic=False` in `ensure_loup_bundle`, the variable
`LOUP_GENERIC=0` and `loupmd validate --specialised` select them, for development and comparisons. The tiled
execution of large cells runs the generic kernels and short-range models only: the `polar1-*` models have no tiled
mode ([Large cells](../guide/large-cells.md)).

The library carries the kernels of the models its release was built from. A model whose symmetric contraction has
no kernels in the library raises `UnsupportedModel`, which names the missing kernel signature.
`loupmd build-kernels MODEL` compiles a kernel plugin for it into `<cache root>/lib`, where the engine finds it
([Command-line reference](cli.md)).

## Models outside the table

A path or a URL outside the table runs when its kernels exist and it is trusted (`sha256=` or `trusted=True`). A
file given by path whose SHA-256 is in the table is recognised as that checkpoint and gets its family and the
training functional of its head (a multi-head file needs `head=`). Any other file has no label, family, licence or training functional; a dispersion correction then needs
`functional=` or `parameters=`.
