# Auto-distillation

`loupgym distill` (and `loupgym.auto_distill(...)`) automatically builds, trains, selects, exports, and verifies a compact **`FastLR-SK`** (`arch=ilmlip`) student potential tailored to your chemical system.

All three data sources (`--structure`, `--md-data`, `--dft-data`) are optional — provide any combination and LoupGYM adapts the pipeline automatically:

```bash
# Structure-only: runs teacher MD across temperatures, labels frames, trains & gates the best student
loupgym distill --teacher omat0-medium --structure cell.cif --speed fast50 --output ./student_out

# Existing MD trajectories + DFT frames (teacher labels MD frames, aligns energy offset to DFT)
loupgym distill --teacher mh1 --head matpes_r2scan --md-data md.extxyz --dft-data dft.extxyz --output ./student_out

# Direct DFT-only training (no teacher needed)
loupgym distill --dft-data dft_train.extxyz --speed balanced --output ./student_out
```

## Pipeline stages

1. **System analysis (`loupgym.distill.analyze`):** Inspects the input structures to classify the system (`ionic_crystal`, `molecular_liquid`, `isolated_molecule`, etc.), detects covalent molecular units and hydrogen bonds, measures neighbor densities, and proposes a tailored candidate pool of `FastLR-SK` student architectures.
2. **Dataset preparation (`loupgym.data`):**
   - If `--structure` is given without enough frames, runs short multi-temperature teacher MD trajectories (plus mild volume strains for periodic crystals) using `loupmd` on GPU to generate diverse training, validation, and held-out test frames.
   - Normalizes energy, forces, and 3×3 stress (auto-detecting `REF_energy`/`energy`, `REF_forces`/`forces`, `SinglePointCalculator`, or custom `--energy-key` / `--forces-key` / `--stress-key`).
   - When both `--teacher` and `--dft-data` are provided, fits per-species E0 alignment between teacher labels and DFT references so both datasets train coherently.
   - Attaches `mol_id` arrays automatically for molecular liquids so net-force and torque losses can be applied.
3. **Candidate tournament (`loupgym.distill.cuda_trainer`):** Trains candidate `FastLR-SK` students in pure CUDA (`build/libloup_lrsk_full_train_v3.so`), benchmarks each candidate's actual inference speed in `libloupmd.so`, and picks the most accurate student that meets the target speed budget.
4. **Optional active learning (`--active-learning-rounds`):** Runs short MD with the winning student, selects high-error/novel frames, relabels them with the teacher, and refines the student.
5. **Export & verification:** Exports the winning student to `student.loup` (`arch=ilmlip`) via `loupmd.export.ilmlip.export_ilmlip_checkpoint`, runs the 3× float32-vs-float64 accuracy gate against `libloupmd.so`, and writes `report.json`.

## Speed presets & budgets

| Preset (`--speed`) | Target MD step cost | Typical architecture |
| :--- | :--- | :--- |
| `fast50` *(default)* | `<= 0.05 µs/atom` (`~0.25 ms` on 5,000 atoms) | Compact 2-layer `FastLR-SK` (`c=(4,4,0)` or `c=(2,2,0)`, `nb=(8,4)`/`(8,8)`) |
| `balanced` | `<= 0.10 µs/atom` | Wider radial/angular basis and higher cutoff |
| `accurate` | `<= 0.25 µs/atom` | Largest candidate basis for maximum fidelity |

| Budget (`--budget`) | Scope |
| :--- | :--- |
| `quick` | Fast screening & shorter training epochs (smoke runs / prototyping) |
| `standard` *(default)* | Full candidate tournament + full training schedule |
| `thorough` | Extended epochs + larger candidate sweep + active learning refinement |

## Verified Phase 2 distilled student benchmarks (RTX 5060 Ti, `sm_120`)

| System | `FastLR-SK` Architecture | Atoms (`N`) | NVE (`µs/at`) | Langevin (`µs/at`) | 3× Gate (`E / F / S`) | Accuracy (`float64`) |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| **SrTiO₃** | `c=(4,4,0)` `t=[16,16,12,8]` `nb=(8,4)` `r=4.6` | 5,000 | `0.0462` | `0.0490` | PASS (`0.02× / 0.17× / 0.06×`) | **960/960 phonons**; test F = `7.28 meV/Å` |
| **Molten Salt** | `c=(4,4,0)` `t=[24,16,8,4]` `nb=(8,4)` `r=6.2` | 3,888 | `0.0451` | `0.0474` | PASS (`0.00× / 0.12× / 0.05×`) | test F = `8.22–8.40 meV/Å` |
| **`n3tbu_v1`** | `c=(4,4,0)` `t=[16,16,8,4]` `nb=(8,4)` `r=(4.2,4.8)` | 4,320 | `0.0464` | `0.0493` | PASS (`0.01× / 0.17× / 0.30×`) | test F = `20.81 meV/Å` |
| **Water** | `c=(2,2,0)` `t=[16,16,16]` `nb=(8,8)` `r=(4.2,5.8)` | 5,184 | `0.0464` | `0.0497` | PASS (`0.11× / 0.31× / 0.59×`) | target F = `19.78 meV/Å` (COM `18.1`, torque `12.8`) |

## Built-in fast dispersion (Options B, C, and D)

Standard engine D3 (`dispersion="d3bj"`) costs ~8.4 ms on 5,000 atoms (~35× a `fast50` student step). Distilled `arch=ilmlip` bundles support three built-in GPU D3(BJ) options (`--dispersion`) that run on a dedicated side stream, use a `C¹` force-shifted cutoff (exact zero energy and force at `r_cut`), and automatically refuse accidental double-counting if `dispersion="d3bj"` is passed on top:

| Option | CLI flag | Method | 5,000-at SrTiO₃ (`ms/eval`) | GPU Parity vs `float64` (`dE / dF / dS`) | Best for |
| :--- | :--- | :--- | ---: | :--- | :--- |
| **B (fixed C6 + tail)** | `--dispersion d3bj` | Force-shifted fixed-C6 D3(BJ) over student neighbour list + live-volume `K/V` tail | `1.104` (`+0.130 ms` vs `0.974` base) | `6.5e-9 / 8.8e-9 / 6.1e-11` | Bulk periodic crystals, salts, liquids (**65× faster** than exact D3) |
| **C (CN-dependent C6 + tail)** | `--dispersion d3bj-cn` | 3-pass coordination-dependent C6 + exact CN chain-rule forces/virial + `K/V` tail | `1.385` (`+0.411 ms`) | `9.2e-9 / 8.9e-9 / 1.6e-10` | Molecules / liquids needing exact local CN-dependent dispersion |
| **D (explicit far pairs)** | `--dispersion d3bj-far --dispersion-rcut 10.0` | On-device spatial cell-bin walk out to `r_disp` (9–12 Å) + optional tail beyond `r_disp` | `3.862` (`r_disp = 9 Å`) | `5.2e-9 / 2.1e-8 / 9.3e-10` | Slabs, interfaces, adsorbates, and fibres/clusters in vacuum (`use_tail=False`) |

Set `LOUP_ILMLIP_DISP=0` at runtime to toggle built-in dispersion off (zero overhead when off). On `n3tbu_v1` (ωB97M-D3BJ, 120-epoch matched A/B retrain), subtracting built-in D3(BJ) during training reduced held-out test force RMSE from `62.17 meV/Å` (baseline) to `57.23 meV/Å` (Option B), `58.65 meV/Å` (Option C, best energy `0.554 meV/at`), and `58.53 meV/Å` (Option D 10 Å).

## Using the distilled student in LoupMD

```python
import loupmd as lm

res = lm.md("cell.cif", model="./student_out/student.loup", temperature_K=300, steps=100_000, output="md_student")
print(res.summary)
```
