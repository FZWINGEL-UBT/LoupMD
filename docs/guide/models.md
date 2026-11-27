# Models

A model is named by a label, a checkpoint path or a URL. LoupMD finds the file, checks that it may be loaded,
exports it once to its own bundle format and runs it in float32 on the engine.

```python
import loupmd as lm

ff = lm.load("omat0-medium")
result = lm.md("structure.cif", model=ff, temperature_K=300, steps=10_000)
```

## Labels

`loupmd models` prints the label table: each label's file, head, training functional, licence, SHA-256 and where
the file was found. `--long` adds the full hash, the release URL and the source of each functional.

| Labels | Checkpoint | Licence |
|---|---|---|
| `mp0-medium`, `mp0-large`, `mp0b3-medium`, `mpa0-medium` | MACE-MP-0, MP-0b3, MPA-0 | MIT |
| `omat0-medium`, `omat0-small` | MACE-OMAT-0 | ASL |
| `matpes-pbe`, `matpes-r2scan` | MACE-MatPES | ASL |
| `mh1-omat`, `mh1-mp`, `mh1-r2scan` | MACE-MH-1, heads `omat_pbe`, `mp_pbe_refit_add`, `matpes_r2scan` | ASL |
| `polar1-s`, `polar1-m`, `polar1-l` | MACE-POLAR-1 | ASL |
| `omol0-xl` | MACE-OMOL-0 extra large | ASL |
| `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium` | MACE-OFF23, OFF23b, OFF24 | ASL |

ASL is the Academic Software License (non-commercial use only). [The model reference](../reference/models.md) has
the full table.

A label fixes its head. A multi-head checkpoint without its own label takes the head by name, either as
`lm.load("mh1", head="omol")` or as the label `mh-1:omol`. There is no default head: a multi-head model without
`head=` is refused with its heads listed, also when its file is given by path. Only `loupmd validate` on a
multi-head file given by path without `--head` picks one itself (`omat_pbe` where the file has it).

## Finding the file

For a label LoupMD looks, in this order:

1. at `checkpoint=` (`--checkpoint PATH`), when given;
2. in the folder named by `LOUPMD_MODELS`;
3. in the cache, `<cache root>/models/`;
4. at the official release URL, only when the download is asked for.

```bash
loupmd models --download omat0-medium
```

`download=True` in Python and `--download` on `loupmd md`, `relax`, `bench` and `validate` do the same when the file
is missing. A download is checked against the table's SHA-256 and is deleted, never loaded, if it differs.

## The trust rule

A MACE checkpoint is a Python pickle and runs code when it is loaded. LoupMD unpickles a file only when:

- it is a label's file and its SHA-256 matches the table, or
- its SHA-256 matches `sha256=` (`--sha256 HEX`), or
- `trusted=True` (`--trusted`) is given.

```python
ff = lm.load("my_model.model", sha256="9b64b4fd...")             # the hash its publisher gives
ff = lm.load("https://example.org/my_model.model", sha256="...", download=True)
ff = lm.load("my_fine_tune.model", trusted=True)                # a file you made yourself
```

A label's file with another hash is refused unless `trusted=True`. A URL without `sha256=` or `trusted=True` is
refused before anything is downloaded. A path is refused without one of the two even when it is a released file;
once trusted, a released file is recognised by its hash and gets its elements and functional from the table. Every
refusal is a `CheckpointTrustError` ([errors](../reference/errors.md)).

For a file outside the table, the elements and heads are read in a separate process when torch and mace-torch are
installed.

## Charge and spin

Two families take the total charge and spin of the cell. MACE-OMOL-0 (`omol0-xl`) embeds them. MACE-POLAR-1
(`polar1-s`, `polar1-m`, `polar1-l`) holds its atomic charges to them: the charges of the two spin channels sum to
(Q + M - 1) / 2 and (Q - M + 1) / 2 for total charge Q and spin multiplicity M. Pass them as `charge=` (in e) and
`spin_multiplicity=` (2S + 1), or `--charge` and `--spin-multiplicity`; the defaults are 0 and 1. Both must be
integers. The export writes them into the bundle, so each setting is its own bundle. Other models refuse them.

```python
ff = lm.load("polar1-m", charge=-1)                        # an anion in its box
ff = lm.load("polar1-m", charge=1, spin_multiplicity=2)    # a radical cation
```

LoupMD's cells are periodic, so a charged molecule is a charged periodic cell, as it is for MACE with `pbc=True`:
the long-range term has no k = 0 Coulomb contribution, which amounts to a uniform neutralising background. Use a
box large enough for the interaction between the images to be negligible. Up to 0.0.2 the POLAR-1 models refused
`charge` and `spin_multiplicity` and ran every cell as a neutral singlet.

## The bundle and its cache

The engine does not read PyTorch checkpoints. At the first evaluation or run, the model is exported to a `.loup`
bundle for the structure's elements. The export runs in a process of its own and needs the export extra
(`pip install 'loupmd[export]'`: torch, mace-torch, cuequivariance, graph_longrange).

### Direct `.loup` bundles

If you already have a `.loup` bundle — for example from [LoupGYM fine-tuning](../loupgym/finetune.md) or
[auto-distillation](../loupgym/distill.md) — pass its path directly to `lm.load` or `lm.md`:

```python
ff = lm.load("finetuned.loup")
res = lm.md("structure.cif", model=ff, temperature_K=300, steps=10_000)
```

Direct `.loup` bundles are binary files (not Python pickles), so they load without PyTorch, skip the export step,
and still validate element coverage (and an optional `sha256=` checksum if you pass one).

Bundles are kept in `<cache root>/bundles/`. A bundle's file name carries a key over the model file's SHA-256, the
elements, the head, charge and spin, and the exporter's own sources, the product-kernel generator included. A
changed exporter never finds an old bundle, and each bundle carries the source of its product kernels. Two processes that need the same bundle export it once: the export runs under a file lock. A
structure with an element the model has no embedding for is refused (`ModelElementError`) before any export.

`loupmd cache list --kind bundles` shows them; `loupmd cache prune --older-than DAYS` (or `--max-size SIZE`) removes
old ones, which are exported again when needed. The cache root is `LOUPMD_CACHE` (default `~/.cache/loupmd`; see [environment](../reference/environment.md)).

## One set of kernels

Every model runs the same kernels (the "generic" kernels of 0.0.1). Version 0.0.1 also had kernels specialised to
MACE-OMAT-0 medium, selected with `LOUP_GENERIC=0` and gated with `loupmd validate --specialised`. They are gone:
the generic kernels have since overtaken them. `LOUP_GENERIC` is now ignored, `loupmd validate --specialised` is
refused, and a bundle that was exported for the specialised kernels is refused with a message to export it again
(the bundle cache does that by itself). Per-atom energies and the tiled execution need a short-range model:
MACE-POLAR-1 has neither.

## Kernel plugins

The library carries the product kernels of the models its release was built from, one per contraction signature. A
model with another signature is refused with `UnsupportedModel`, whose message names the signature and the command
that builds it:

```bash
loupmd build-kernels my_model.model
```

`loupmd build-kernels MODEL` generates the kernels and compiles them with `nvcc` into a plugin,
`<cache root>/lib/loupmd-kernels-<signature>.so`. The engine finds it by its signature, and the model then loads as
any other. It needs the CUDA toolkit (13.x), the export extra and the kernel sources (a source checkout, or a wheel
that ships them). `--arch sm_XY` compiles for another GPU than GPU 0; `--force` compiles again. The generator loads the
checkpoint, so the trust rule of every other command applies: a file outside the label table needs `--sha256 HEX`
or `--trusted`. `LOUP_KERNEL_PATH` adds folders the engine searches for plugins. A plugin built for 0.0.1 is not
loaded by a later library (the kernel interface changed): build it again.

## The accuracy gate

`loupmd validate --model LABEL` compares LoupMD's energy, forces and stress with MACE in float64. A quantity passes
when LoupMD's deviation from MACE float64 is at most three times MACE float32's own deviation on the same structure,
or below a floor:

| Quantity | Passes when | Floor |
|---|---|---|
| energy | dE <= max(3 dE32, floor) | 0.01 meV/atom |
| forces (largest component) | dF <= max(3 dF32, floor) | 1e-5 eV/A |
| stress (largest component) | dS <= max(3 dS32, floor) | 1e-7 eV/A^3 |

```bash
loupmd validate --model omat0-medium --structure structure.cif --json gate.json
```

- `--structure FILE` gates rattled copies of your structure (`--copies`, default 2; `--rattle`, default 0.05 A).
- `--set` chooses a shipped set instead: `oxide` (default), `organic` or `crystals`. The NbCl5, NaCl and NaNbO3
  cells of `oxide` and `crystals` are not shipped; put them in a folder named by `LOUPMD_BENCH_STRUCTURES`.
  `organic` needs no file.
- `--ref-device cpu` computes the references on the CPU, for models whose float64 pass does not fit the GPU.
- `--capacity tiled` or `wavefront` gates the tiled execution.
- The references are cached in `<cache root>/gate_refs/`; `--no-ref-cache` computes them again.

For the MACE-POLAR-1 models the stress reference is the strain derivative of MACE's float64 energy, taken by
central differences, and LoupMD's stress agrees with it. mace-torch up to 0.3.16 returns a stress for these models
that lacks the cell dependence of the long-range term: on a 648-atom water box with `polar1-s` its diagonal
components were 0.02 GPa (about 1%) off. mace-torch 0.3.17 has the term, and its stress agreed with LoupMD's on
that box. To compare LoupMD's stress of a POLAR-1 model with MACE's own, use mace-torch 0.3.17 or newer.

The gate needs the export extra. It exits with code 1 when a quantity fails. Before an MD run, `check="gate"`
(`--check gate`) runs the same rule on rattled copies of the run's structure and stops with `GateFailed` if it
fails. With `"allow_unavailable"` a gate whose references cannot be computed is recorded as unavailable and the run
goes on.

## Precision

`lm.load(..., precision="float32")` is the only setting, and the default. LoupMD computes forces in float32 with
float64 accumulation. `precision="float64"` is refused with the reason ([not in LoupMD](not-in-loupmd.md)). For
float64 reference values, run MACE itself with `default_dtype="float64"`.
