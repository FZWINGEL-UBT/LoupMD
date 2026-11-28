# LoupGYM CLI and Python API

## Command line (`loupgym`)

Invoke via `loupgym COMMAND` or `python -m loupgym COMMAND`.

### `loupgym finetune`

Fine-tune any LoupMD foundation MACE model and output a verified `.loup` bundle.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--model MODEL` | *(required)* | Foundation label (`omat0-small`, `mh1`, `polar1-s`, …), `.model` path, or `.loup` bundle path |
| `--train FILE` | *(required, repeatable)* | Training frames (`.extxyz`, `.xyz`, `.traj`) |
| `--valid FILE` | auto-split | Validation frames (repeatable) |
| `--test FILE` | auto-split | Held-out test frames for zero-shot vs fine-tuned evaluation (repeatable) |
| `--replay-data FILE` | none | Replay frames mixed into training batches to prevent forgetting (repeatable) |
| `-o`, `--output PATH` | `./finetune_out` | Output directory, `.loup` path, or `.model` path |
| `--backend {cuda,pytorch}` | `cuda` | `cuda` (pure-CUDA `libloupmd_ft.so`, default) or `pytorch` |
| `--strategy {auto,full,top,readout,lora}` | `auto` | Fine-tuning strategy (`lora` requires `--backend pytorch`) |
| `--head NAME` | none | Head name for multi-head models (e.g. `matpes_r2scan`) |
| `--epochs N` | `30` | Number of training epochs |
| `--lr FLOAT` | `5e-4` | Peak learning rate for readout/head parameters |
| `--trunk-lr-mult FLOAT` | `0.25` | Learning-rate multiplier for trunk parameters |
| `--anchor-weight FLOAT` | `1e-2` | L2 anchor regularization toward pretrained weights |
| `--replay-weight FLOAT` | `0.25` | Relative loss weight on `--replay-data` |
| `--energy-weight FLOAT` | `1.0` | Energy loss weight (boosted 20× in final 20% of epochs) |
| `--force-weight FLOAT` | `100.0` | Force loss weight |
| `--stress-weight FLOAT` | auto | Stress loss weight (`--backend pytorch` only) |
| `--batch-size N` | auto | Training batch size |
| `--grad-accum N` | auto | Gradient accumulation steps |
| `--keep-e0` | off | Skip least-squares per-species E0 alignment |
| `--charge INT` | none | Total cell charge (`polar1-*` and `omol0-xl`) |
| `--spin-multiplicity INT` | none | Spin multiplicity `2S + 1` (`polar1-*` and `omol0-xl`) |
| `--energy-key`, `--forces-key`, `--stress-key` | auto | Custom ASE `info`/`arrays` keys |
| `--no-verify` | off | Skip post-training parity / gate verification |
| `--device N` | `0` | CUDA GPU index (`-1` for CPU with `--backend pytorch`) |
| `--seed N` | `42` | Random seed |
| `--quiet` | off | Suppress progress output |

### `loupgym distill`

Automatically distill a fast `FastLR-SK` (`arch=ilmlip`) student potential.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--teacher MODEL` | none | Teacher foundation label, `.model` path, or `.loup` bundle |
| `--head NAME` | none | Teacher head name for multi-head models |
| `--structure FILE` | none | Starting structure(s) for automatic teacher MD sampling (repeatable) |
| `--md-data FILE` | none | Pre-existing MD trajectory frames to label/train on (repeatable) |
| `--dft-data FILE` | none | Reference DFT frames with energy/forces (repeatable) |
| `-o`, `--output PATH` | `./distill_out` | Output directory or `.loup` path |
| `--speed {fast50,balanced,accurate}` | `fast50` | Target MD speed preset |
| `--budget {quick,standard,thorough}` | `standard` | Compute budget for candidate search & training |
| `--temps K [K ...]` | `300 600` | Temperatures (K) for automatic teacher MD sampling |
| `--dispersion {off,d3bj,d3bj-cn,d3bj-far}` | `off` | Built-in force-shifted D3(BJ): Option B (`d3bj`), Option C (`d3bj-cn`), or Option D (`d3bj-far`) |
| `--dispersion-functional NAME` | `pbe` | D3(BJ) functional parameters (`pbe`, `r2scan`, `wb97m`, `b3lyp`, `pbe0`) |
| `--dispersion-rcut FLOAT` | `10.0` | Explicit far-pair cutoff in Å for `--dispersion d3bj-far` |
| `--active-learning-rounds N` | `0` | Optional student MD active-learning refinement rounds |
| `--device N` | `0` | CUDA GPU index |
| `--seed N` | `42` | Random seed |
| `--quiet` | off | Suppress progress output |

---

## Python API

`import loupgym as lg` exposes the functions and classes below lazily (importing `loupgym` does not import `torch`).

### Fine-tuning

::: loupgym.finetune.core.finetune

::: loupgym.finetune.core.FinetuneResult

::: loupgym.finetune.cuda_backend.finetune_cuda

::: loupgym.finetune.cuda_backend.CudaMaceTrainer

### Auto-distillation

::: loupgym.distill.auto.auto_distill

::: loupgym.distill.auto.DistillResult

### Shared utilities

::: loupgym._lib.find_native_lib

::: loupgym.dispersion_d3.fit_fixed_d3bj

::: loupgym.dispersion_d3.fit_cn_d3bj

::: loupgym.dispersion_d3.transform_labels_for_d3
