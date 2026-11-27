# LoupGYM

**LoupGYM** (`loupgym`) is LoupMD's companion training package for **pure-CUDA foundation MACE fine-tuning** and **automatic student distillation**.

It is intentionally kept separate from `loupmd`: `loupmd` stays a pure MD and inference runtime with zero training entry points, while `loupgym` handles adapting foundation potentials or distilling ultra-fast student models and outputs verified `.loup` bundles that load directly into `loupmd`.

```python
import loupmd as lm
import loupgym as lg

# 1. Pure-CUDA foundation fine-tuning (zero PyTorch in training or inference)
ft = lg.finetune(model="omat0-small", train_data="dft.extxyz", output="./ft_out", epochs=30)

# 2. Or auto-distill a fast student from any foundation teacher
dist = lg.auto_distill(teacher="omat0-medium", structure="cell.cif", speed="fast50", output="./student_out")

# 3. Run MD directly on the resulting .loup bundle (no PyTorch needed)
run = lm.md("cell.cif", model=ft.bundle_path, temperature_K=300, steps=10_000, output="md_run")
```

## Two workflows

| Workflow | Command / Function | When to use it | Output |
| :--- | :--- | :--- | :--- |
| **Foundation fine-tuning** | `loupgym finetune` / `lg.finetune(...)` | Adapt any of the 18 LoupMD foundation models (`omat0`, `mp0`, `mp0b3`, `mpa0`, `matpes`, `mh1`, `off23`/`off24`, `omol0-xl`, `polar1-s/m/l`) to your DFT data with minimal frames | Verified `.loup` bundle (`arch=generic`, same architecture & speed as base model) |
| **Auto-distillation** | `loupgym distill` / `lg.auto_distill(...)` | Distill a compact, system-specific `FastLR-SK` student (`< 0.05 µs/atom` MD) from a foundation teacher and/or your own MD/DFT data | Verified `.loup` bundle (`arch=ilmlip`) + PyTorch student checkpoint |

## Building the native CUDA training libraries

LoupGYM's pure-CUDA trainers compile into separate shared libraries that are never linked into `libloupmd.so`:

```bash
# Pure-CUDA foundation MACE fine-tuning library (libloupmd_ft.so)
make -j4 mace-ft-lib CUDA_ARCH=sm_120

# Pure-CUDA FastLR-SK student distillation library (build/libloup_lrsk_full_train_v3.so)
make -j4 lrsk-train-lib CUDA_ARCH=sm_120
```

Replace `sm_120` with your GPU's compute capability (`sm_80` for A100, `sm_86` for RTX A4000 / 30-series, `sm_89` for RTX 4000 Ada / L4 / 40-series, `sm_120` for RTX 50-series).

When `loupgym` runs, [loupgym/_lib.py](https://github.com/FZWINGEL-UBT/LoupMD/blob/release/v1/loupgym/_lib.py) locates the compiled libraries automatically (or honors `LOUP_FT_LIB` and `LOUP_LRSK_FT_LIB`; see [environment variables](../reference/environment.md)).

## Next steps

- [Fine-tuning foundation models](finetune.md) — strategies, all 6 families, pure-CUDA engine design, and benchmarks.
- [Auto-distillation](distill.md) — system analysis, flexible input combinations, presets, and Phase 2 results.
- [LoupGYM CLI and Python API](reference.md) — full command-line flags and Python signatures.
