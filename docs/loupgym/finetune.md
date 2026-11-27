# Fine-tuning foundation models

`loupgym finetune` (and `loupgym.finetune(...)`) adapts any LoupMD foundation MACE model to your reference energies and forces and writes a verified `.loup` bundle.

By default (`backend="cuda"`), the entire training and evaluation loop runs in **`libloupmd_ft.so`** — a pure C++/CUDA double-backprop engine with **zero PyTorch** in the training loop.

```bash
# Default pure-CUDA backend (zero PyTorch in training; writes verified finetuned.loup)
loupgym finetune --model omat0-small --train train.extxyz --output ./ft_out --epochs 30

# Multi-head (MH-1), long-range (POLAR-1), and charged/open-shell (OMOL-0) examples
loupgym finetune --model mh1 --head matpes_r2scan --train solid.extxyz --output ./ft_mh1
loupgym finetune --model polar1-s --train water.extxyz --strategy top --output ./ft_polar
loupgym finetune --model omol0-xl --train mol.extxyz --charge 0 --spin-multiplicity 1 --output ./ft_omol
```

## Supported model families

The pure-CUDA trainer supports all 6 foundation model families (all 5 generic layer kinds `0..4`, all readout types `0/1/2`, ZBL pair repulsion, and `PolarLR` k-space electrostatics):

| Family | Layer kind | Foundation labels | Notes |
| :--- | :--- | :--- | :--- |
| **Density** | `kind 2 / 3` | `omat0-small`, `omat0-medium`, `mp0b3-medium`, `mpa0-medium`, `matpes-pbe`, `matpes-r2scan` | Full or top-layer fine-tuning |
| **Residual** | `kind 0 / 1` | `mp0-medium`, `mp0-large` | Scale-invariant AdamW handles `.loup` folded `1/(avg_num_neighbors * sqrt(C))` weights automatically |
| **Multi-head** | `kind 4` (SiLU+sigmoid gate) | `mh1-omat`, `mh1-mp`, `mh1-r2scan` (or `--model mh1 --head <name>`) | Active head exported into `.loup` and fine-tuned directly |
| **OFF** | `kind 0 / 1` | `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium` | Organic / molecular force fields |
| **OMOL-0** | `kind 4`, 3 layers (50.7M params) | `omol0-xl` | Pass `--charge` and `--spin-multiplicity`; `--strategy top` recommended |
| **POLAR-1** | `kind 4` + `PolarLR` (`has_lr=1`) | `polar1-s`, `polar1-m`, `polar1-l` | Exact `PolarLR` k-space electrostatics + local trunk double-backprop; `--strategy top` recommended |

## How the pure-CUDA engine works

`libloupmd_ft.so` (`src/train/mace_ft/`) operates directly on `LOUPC001` `.loup` bundles:

1. **Unfolded `.loup` loader & writer (`mace_ft_bundle.cuh`):** Reads all weights, spline radial tables, ZBL parameters, `PolarLR` parameters, and product kernel sources straight into GPU memory without PyTorch, and writes the best EMA weights back to `finetuned.loup`.
2. **Exact two-phase force double-backprop (`mace_ft_train.cu`, `mace_ft_gconv.cuh`, `mace_ft_kernels.cuh`):**
   - **Pass 1 (Forward `t = 0..L-1`):** Evaluates node embeddings, spherical harmonics (`sh16`), radial splines, Clebsch–Gordan convolutions, symmetric products, readouts, ZBL, and optional `PolarLR`.
   - **Pass 2 (First backward `t = L-1..0`):** Computes exact atomic forces `F = -dE/dR` (including `PolarLR` forces `F_lr` and `dh_lr`).
   - **Phase 3a (Reverse of Pass 2, `t = 0..L-1`):** Propagates force-loss tangents (`bar_F -> bar_edge_dvec -> tan_*`) using exact `sh16_jvp` and second-order layer/gate/readout kernels, accumulating force-loss weight gradients.
   - **Phase 3b (Reverse of Pass 1, `t = L-1..0`):** Propagates energy-loss adjoints (`bar_E`) and `bar_E * dh_lr` back through the trunk, accumulating total `d(L_E + L_F)/dW`.
3. **NVRTC Dual-number product HVP (`mace_ft_prod_jit.cuh`):** Mechanically lifts each layer's stored `L<t>.prod_src` kernel to `Dual{val, dot}` arithmetic via NVRTC (`loup::jit::build`, cached on disk), providing exact forward-over-reverse Hessian–vector products for every product signature across all 18 foundation models.
4. **Scale-invariant GPU AdamW + Anchor + EMA (`mace_ft_kernels.cuh`):**
   - Automatically rescales each parameter tensor's learning rate and gradient norm contribution by its initial RMS weight scale (`rms / 0.1` when `rms < 0.1`), compensating for `.loup` constant folding of `1 / (avg_num_neighbors * sqrt(C))`.
   - Applies L2 anchor regularization (`--anchor-weight`) toward the pretrained weights to prevent catastrophic forgetting, clips global gradient norm (`10.0`), and tracks an Exponential Moving Average (`decay_cap = 0.985`).
5. **Active-head E0 alignment & 2-stage schedule:** Fits per-species atomic energy offsets (`E0`) by least squares on the training split before and after training, uses linear warmup + cosine LR decay, and boosts the energy loss weight `20x` in the final 20% of epochs.
6. **Parity verification:** After saving `finetuned.loup`, evaluates test frames with `libloupmd.so` (`LoupEngine`) and verifies exact energy and force agreement (`dF_max <= 1e-3 eV/Å`).

## Verified GPU benchmarks (RTX 5060 Ti, `sm_120`)

Every family was verified end-to-end with `loupgym.finetune(..., backend="cuda")` and checked for `libloupmd.so` parity:

| Family | Model | Strategy (Trainable Params) | Zero-shot F (meV/Å) | Fine-tuned F (meV/Å) | Fine-tuned E (meV/at) | `libloupmd.so` Parity (`max\|dF\|`) | Pure-CUDA Train Time |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Density** | `omat0-small` *(zero `torch`)* | `full` (554,784) | 52.04 | **8.82** | 0.467 | `2.41e-5 eV/Å` (PASS) | **0.62 s** (10 ep) |
| **Residual** | `mp0-medium` | `full` (544,544) | 138.35 | **37.19** | 0.410 | `5.70e-6 eV/Å` (PASS) | **2.21 s** (10 ep) |
| **Multi-head** | `mh1` (`matpes_r2scan`) | `full` (6,432,288) | 48.15 | **17.91** | 0.474 | `1.28e-5 eV/Å` (PASS) | **2.32 s** (10 ep) |
| **OFF** | `off24-medium` | `full` (812,704) | 21.67 | **19.82** | 0.296 | PASS | ~1 s (8 ep) |
| **OMOL-0** | `omol0-xl` (`q=0, s=1`) | `top` (17.19M / 50.68M) | 43.38 | **23.58** | 0.232 | `2.31e-5 eV/Å` (PASS) | **10.09 s** (6 ep) |
| **POLAR-1** | `polar1-s` (81-at H2O) | `top` (5.19M) | 122.57 | **44.53** | 0.873 | `9.80e-6 eV/Å` (PASS) | ~3 s (8 ep) |

On `omat0-small` (40-atom SrTiO3, 10 epochs), pure-CUDA wall time is **2.08 s** (`0.62 s` training loop, zero `torch` imported) vs **44.21 s** with PyTorch + CuEquivariance (**21.2× faster** wall-clock).

## Strategies & anti-forgetting

Choose `--strategy` based on dataset size and model size (or leave `--strategy auto`):

| Strategy | Trainable parameters | Recommended for |
| :--- | :--- | :--- |
| `auto` *(default)* | Selects `readout` (< 20 frames), `top` (large models / `PolarMACE` / `omol0-xl`), or `full` | General use |
| `full` | All linear, skip, product, gate, embedding, and readout tensors | Medium/large datasets (≥ 50 frames) on small/medium models |
| `top` | Final interaction layer + all readouts (`trunk_lr_mult = 0.25`) | Large models (`omol0-xl`, `polar1-*`, `mh1`) or limited data |
| `readout` | Readout layers only | Very small datasets (< 20 frames) or quick energy/force calibration |
| `lora` | Equivariant low-rank adapters (`--lora-rank`, `--lora-alpha`), merged back before export | Available with `--backend pytorch` when exporting a `.model` checkpoint |

To prevent forgetting outside your fine-tuning domain:
- Keep `--anchor-weight 1e-2` (default) to pull weights gently toward the pretrained checkpoint.
- Optionally pass `--replay-data replay.extxyz --replay-weight 0.25` to mix general-domain frames into every epoch.

## Pure-CUDA vs PyTorch backend

| Feature | `--backend cuda` *(default)* | `--backend pytorch` |
| :--- | :--- | :--- |
| Training engine | `libloupmd_ft.so` (pure C++/CUDA, zero `torch` when given a `.loup` bundle) | PyTorch (+ optional CuEquivariance) |
| Speed | **~20× faster** wall-clock on RTX 5060 Ti | Standard PyTorch speed |
| Output | Verified `finetuned.loup` + `report.json` | `finetuned.model` + exported `finetuned.loup` + 3× float64 gate |
| Stress loss / LoRA / CPU | Not needed for energy+force fine-tuning (use `--backend pytorch` if you need stress loss, `--strategy lora`, `.model` export, or CPU training) | Supported |

## Using the fine-tuned bundle in LoupMD

The output `finetuned.loup` loads directly in `loupmd` without PyTorch or re-export:

```python
import loupmd as lm

ff = lm.load("./ft_out/finetuned.loup")
res = lm.md("cell.cif", model=ff, temperature_K=300, steps=50_000, output="md_ft")
```
