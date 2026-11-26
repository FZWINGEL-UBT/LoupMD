# Loup-S Phase 1 & Phase 2 Distillation Results (Architecture Benchmark Reference)

Measured on `loup-s-train-engine` (RTX 5060 Ti `sm_120` and RTX 4000 Ada `sm_89`). All models use the fast SOAP + gauge-centered $E_i^{\text{teacher}}$ training pipeline and are evaluated natively in LoupMD (with `C = 16` native engine support added and 3× float32-vs-float64 gated across all forced variants).

## 1. SrTiO₃ (Perovskite Oxide, 3 species: Sr, Ti, O; Teacher: MACE-MH-1 `omat_pbe`, `4.594 µs/atom-step` on 5,000 atoms, 5060 Ti)

Fidelity bar: overall & per-species force RMSE $\le 2\%$, energy RMSE $\le 1$ meV/atom, 960/960 phonons within max(3%, 0.1 THz), matched NPT lattice & NVE drift/RMS.

| Model | Irreps / Conv | `R` | $r_{\max}$ (Å) | Params | `common_tgt60` F% | `E` (meV/at) | `common_hard40` F% | §4 MD Status | 5,000-at Step (5060 Ti) | µs/at | Speedup vs MH-1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| MACE-MH-1 (teacher) | 512 / 128 | 64 | 6.0 | ~4.7M | 0.00% | 0.000 | 0.00% | ref | 22.97 ms | 4.594 | 1.00× |
| `sto_s64l0_c55_allgreen` | `64x0e` | 64 | 5.5 | 171k | **0.59% PASS** | 0.015 | **17.4%** (short-contact probes) | **960/960 phonons PASS, NPT cell −0.00002 Å PASS** | 2.18 ms | 0.436 | 10.5× |
| `sto_s64r8_c55` | `64x0e` | 8 | 5.5 | 137k | **0.75% PASS** | 0.016 | — | **960/960 phonons PASS** | 1.745 ms | 0.349 | 13.17× |
| `sto_s64r8_c50` | `64x0e` | 8 | 5.0 | 137k | **0.75% PASS** | 0.016 | — | label PASS | 1.811 ms | 0.362 | 12.68× |
| `sto_c64_m32_r8_c50` | `64x0e` / `32` conv | 8 | 5.0 | 126k | **0.80% PASS** | 0.017 | — | label PASS | 1.313 ms | 0.263 | **17.49×** |
| `sto_c64_m16_r8_c50` | `64x0e` / `16` conv | 8 | 5.0 | 120k | **0.86% PASS** | 0.021 | — | 3× gate PASS (all 6 variants) | 1.466 ms | 0.293 | 15.67× |
| **`sto_s32r8_c50`** | **`32x0e`** | **8** | **5.0** | **46k** | **0.94% PASS** (1.18% allgreen) | **0.022** | 54.2% (**15.97%** allgreen) | **3× gate PASS; 960/960 phonons PASS, lat −0.024% PASS** | **1.126 ms** | **0.225** | **20.41×** |
| **`sto_s16r8_c50`** | **`16x0e`** | **8** | **5.0** | **21k** | **1.24% PASS** | **0.021** | — | 3× gate PASS (0.75×, all 6 variants) | **1.004 ms** | **0.201** | **22.87×** |
| `sto_16v1_r8_c50` | `16x0e+16x1o` | 8 | 5.0 | 27k | 1.84% | 0.391 | 16.09% | 3× gate PASS (1.37×, all 6 variants) | 1.556 ms | 0.311 | 14.77× |

Key physics findings on SrTiO₃:
- Pure scalar `l = 0` messages (`64x0e`, `32x0e`, even `16x0e` with only **21,200 parameters**) easily clear `< 1.3%` force RMSE and `~0.02 meV/at` energy RMSE on 3-species crystal bulk.
- `R = 8` radial head costs zero accuracy vs `R = 64` while dramatically shrinking Layer 0 and `C0` GEMMs.
- Training speed in native C++/CUDA `loupstrain_cli` (`s64l0`, 300 frames, SOAP + $E_i^T$): **2.87 s GPU / ~3.5 s wall** (vs 30–140 GPU-hours in stock `mace-torch`).

## 2. Molten Salt (Li/Na/K/Cl at 1200–2160 K, 4 species; Teacher: MACE-MH-1 `omat_pbe`, `4.68 µs/atom-step` on 5,832 atoms, RTX 4000 Ada)

Fidelity bar: overall & per-species force RMSE $\le 2\%$, 4-species self-diffusion within teacher 4-seed range, 1800 K & 2160 K NVT stability.

| Model | Irreps | `R` | $r_{\max}$ (Å) | Params | `valid` F% | `test` F% | `teacher_hard` F% | 3× Gate (Ada) | 5,832-at Step (Ada) | µs/at | Speedup vs MH-1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| MACE-MH-1 (teacher) | 512 / 128 | 64 | 6.0 | ~4.7M | 0.00% | 0.00% | 0.00% | ref | 27.3 ms | 4.68 | 1.00× |
| **`s_64v1_agn_r8_c60_ei_v3`** | `64x0e+64x1o` | 8 | 6.0 | 188k | **1.37% PASS** | **1.41% PASS** | **1.58% (stu4)** | PASS | 2.484 ms | 0.426 | **10.99×** |
| **`s_48v1_agn_r8_c60_ei`** | `48x0e+48x1o` | 8 | 6.0 | 131k | **1.47% PASS** | **1.46% PASS** | 1.67% | **PASS (0.34× F, 0.10× S)** | 2.062 ms | 0.354 | **13.24×** |
| **`s_32v1_agn_r8_c60_ei_wd`** | `32x0e+32x1o` | 8 | 6.0 | 70k | **1.53% PASS** | **1.57%** (`K=1.90%`) | 1.70% | **PASS (0.43× F, 0.29× S)** | **1.360 ms** | **0.233** | **20.07×** |
| `s_16v1_agn_r8_c60_ei` | `16x0e+16x1o` | 8 | 6.0 | 30k | 2.14% | 2.14% | 2.24% | **PASS (0.59× F, 0.25× S)** | **1.008 ms** | **0.173** | **27.07×** |
| `s_16v1_agn_r64_c60_ei` | `16x0e+16x1o` | 64 | 6.0 | 50k | 2.14% | 2.14% | 2.28% | — | 1.970 ms | 0.338 | 13.86× |
| `s_32s0_agn_r8_c60_ei` | `32x0e` | 8 | 6.0 | 53k | 2.26% | 2.30% | 2.27% | — | **0.870 ms** | **0.149** | **31.37×** |

Key physics findings on Molten Salt:
- Vector messages (`l = 1`, `Nx0e+Nx1o`) are strongly beneficial over scalar `l = 0` (`32v1` 1.53% vs `32s0` 2.26%) for capturing ionic polarization/dipole correlations across 4 species.
- `R = 8` matches `R = 64` identically (`2.14%` vs `2.14%` at `C = 16`) while running **1.95× faster** (`0.173` vs `0.338 µs/at`)!
- `s_64v1_agn_r8_c60_ei_v3` (`11.0×` MH-1) passed **4/4 species diffusion (Li, Na, K, Cl)** and survived 1800 K + 2160 K NVT.
- `s_32v1_agn_r8_c60_ei_wd` (`70k params`, **`20.1×` MH-1**) with `weight_decay = 1e-5` **survived 1200 K NPT (2 seeds) + 1800 K + 2160 K NVT** (curing the unregularized 2160 K blowup), with 2/4 diffusion in teacher 4-seed range (`Li +11.2%`, `Na -3.1%`, `Cl +8.9%`, `K +21.7%`).
- `s_48v1_agn_r8_c60_ei` (`131k params`, `13.2×` MH-1) hit 3/4 diffusion in range (`K +0.5%`, `Li +8.7%`, `Na -5.6%`, `Cl +10.0%`).

## 3. Flexible Liquid Water (216 H₂O = 648 atoms; Teacher: MACE-OFF24_medium, `2.052 µs/atom-step` on 5,184 atoms, Ada)

Fidelity bar: overall & per-species force RMSE $\le 2\%$, molecular COM force & torque RMSE $\le 2\%$, matched 2×40 ps NPT density / RDF / O self-diffusion, 0.5 fs NVE.

- **Scalar `l = 0` fails badly on water** (`64x0e` even with `R = 64`, `r_max = 6.0`, `zbl = False` stalls at loss `0.165` vs `0.009` for `64v1` — ~18× worse loss; vector `l = 1` dipoles are essential for H-bonding and molecular torques).
- **Radial head `R = 64` + `r_max = 6.0` + `zbl = False` (matching OFF24-M!) are critical on water:**
  - `w_32v1_agn_r8_nozbl_c55` (`R = 8, 5.5 Å`, 52k): `4.24%` F, `COM 7.71%`, `TRQ 7.55%` (`0.537 µs/at` on 648 at)
  - `w_32v1_agn_r64_nozbl_c60` (`R = 64, 6.0 Å`, 85k): `2.62%` F, `COM 6.48%`, `TRQ 5.57%` (`0.709 µs/at` on 648 at)
  - `w_64v1_agn_r8_c60_v3` (`zbl = True, bs = 4`): `2.86%` F, `COM 5.78%`, `TRQ 4.67%` — ALREADY passed matched 2×40 ps NPT **O diffusion (`-2.1%`, 8.6× inside teacher spread)** and H/O displacements, missing density by only `0.0009 g/cm³` (ZBL kink on O–H bond caused `H-O` RDF max error `0.26`).
  - `w_64v1_agn_r64_nozbl_c60` (`R = 64, 6.0 Å, zbl = False, bs = 2`, 213k): dropped training loss **3× further to `0.0091`** (re-running cleanly with OOM-safe E0 calibration + full §4 MD on `murphy`).

## 4. Measured `nsys` Kernel Census at `C = 64 -> 32 -> 16` (5,000-atom SrTiO₃, RTX 5060 Ti, `nsys --cuda-graph-trace=node`)

Exact per-step GPU kernel times averaged over 20 300 K Langevin steps (after warmup + autotune):

| Kernel Group | `sto_s64r8_c50` (`64x0e`) | `sto_s32r8_c50` (`32x0e`) | **`sto_s16r8_c50` (`16x0e`)** | Share of `s16` | Scaling `64 -> 32 -> 16` |
|---|---:|---:|---:|---:|---|
| **Layer-1 conv** (`gconv_bwd_L0` + `gconv_fwd_L0`) | 453.4 + 225.0 = **678.4 µs** | 256.0 + 118.2 = **374.2 µs** | 227.2 + 98.7 = **325.9 µs** | **34.6%** | 1.81× then only **1.15×** (50% idle lanes!) |
| **Layer-0 edge** (`first_bwd_edge` + `first_fwd`) | 233.0 + 88.3 = **321.3 µs** | 234.4 + 89.0 = **323.4 µs** | 227.1 + 88.1 = **315.3 µs** | **33.5%** | **1.00×** (invariant; 85% idle lanes on `J-1=9`) |
| **All GEMMs** (`hgemm2f` + `sgemm*` + `cutlass`) | 405.0 + 16.0 = **421.0 µs** | 194.4 + 22.3 = **216.7 µs** | 147.4 + 18.1 = **165.5 µs** | **17.6%** | **32 calls/step** of `128×64` `hgemm2f` on 16-col matrices! |
| **Products** (`gprod_ro1` + `gprod_bwd` + `gprod_fwd`) | 115.6 + 81.9 + 35.7 = **233.2 µs** | 41.9 + 33.8 + 15.5 = **91.2 µs** | 30.9 + 15.6 + 12.6 = **59.1 µs** | **6.3%** | 2.56× then **1.54×** (50% idle lanes) |
| **Compaction + rebuild + Langevin + skin/guards** | **~76 µs** | **~77 µs** | **~77 µs** | **8.2%** | Hard MD/neighbor floor (compact 37 µs, rebuild 15 µs, Langevin 17 µs) |
| **GPU kernel sum / Unprofiled wall step** | 1,730 / **1,811 µs** | 1,083 / **1,126 µs** | 943 / **1,004 µs** | **100%** | **`12.7×` -> `20.4×` -> `22.9×` MH-1** |

Even though `sto_s16r8_c50` has only 21k params and already hits **22.9× MH-1** (`1.80×` faster than `s64r8`, clearing the `LOUP_S_DESIGN.md` §4 stop rule), **91.8% of its GPU time (`866 / 943 µs`) is still neural-network kernel overhead** (the true neighbor + integrator floor is only `77 µs` = `0.015 µs/atom-step` = **~300× MH-1**!):

1. **Channel-invariant Layer-0 + neighbor/compaction floor (~40% of step, ~0.41 ms):** `first_fwd_kernel`, `first_bwd_edge_kernel`, compaction, and neighbor rebuild do not shrink with `C`. At `R = 8` (`J = 10` columns), `first_fwd_kernel` still launches `64` threads/atom (only 9 accumulate: 86% idle) and `first_bwd_edge_kernel` uses 32-lane warps with only 5 active lanes (84% idle).
2. **50% idle warp lanes across all `C = 16` conv and product kernels:** `gconv_fwd_L0`, `gconv_bwd_L0`, `gprod_fwd`, `gprod_bwd`, and `gprod_ro1_fused` all round `blockDim` up to 32 threads with `active = (ch < 16)`.
3. **Tiny 16-column GEMMs + ~30 separate kernel launches / global-memory round-trips:** At `C = 16`, all weights (`C0`, `U`, `V_l`, `K`, `P3`) total `< 20 KB` and fit inside shared memory/registers, yet the generic path writes/re-reads every intermediate node tensor (`S0, m0, h, x, mid, m1` and gradients) across separate launches.
4. **Why non-MACE architectures could go even faster (sub-`0.10 µs/at`, `50×+` MH-1):**
   - Eliminating MACE's 2nd message-passing hop (or replacing it with a strictly local 1-hop many-body/ACE/SNAP/NEP/Cartesian-tensor descriptor + compact MLP, or dual-cutoff short many-body `~4.0–4.2 Å` + fast 2-body/dipole outer shell `6.0 Å`) cuts active edges by 2–3× and removes the entire Layer-1 forward/backward pass + inter-layer communication.
   - Replacing `l <= 3` spherical harmonics + high-degree Clebsch-Gordan `P3` contractions with low-cost Cartesian invariants/vectors (rank-0/1/2 moment tensors / MTP / Allegro / NequIP-Cartesian / Painn / NEP) avoids both the 16-component `Y_lm` tables and the heavy `P3` polynomial backward.
   - For molecular liquids (water) and ionic melts (salt), the bottleneck is long-range/medium-range dipole-dipole & H-bond orientation (`l = 1` + `6.0 Å` radial resolution) rather than high-order `l = 3` angular body order — an architecture combining a **ultra-cheap short-range bonded/repulsive core (`~3.5–4.2 Å`) + cheap rank-1 vector/dipole environment (`6.0 Å`)** directly targets the real physics bottleneck at a fraction of MACE's FLOPs and bytes!
