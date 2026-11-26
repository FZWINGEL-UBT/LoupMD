# Optimisers for Loup-S training: Adam, Muon and SOAP

This note explains why the Loup-S training recipes on `loup-s-train-engine` use SOAP, and what we saw with Adam
and Muon before it. It is a summary of the records in `OPTIMISATION_LEDGER.md` (entries `ft-recipe`, `ft-muon`,
`ft-muon-no-l2`, `ft-dpa4-hybridmuon`, `ft-soap`, `ft-recipe-b1`, `ft-recipe-soap-betas`,
`ft-recipe-muon-momentum`, `ft-recipe-muon-wd`, `ft-soap-kern`) and `STATUS_AND_PROFILE.md`. Nothing here was
measured again for this note.

## Summary

- **Adam** (mace's default, with AMSGrad) was limited by the optimisation, not by the data: after 60 epochs the
  validation force error was about 12 meV/A and it was still falling at 240 epochs (8.1).
- **Muon** on the weight matrices reached 7.3 to 8.1 meV/A in the same 60 epochs, for 3% more time per step at
  batch 4.
- **SOAP** on the same matrices reached 5.7 to 6.2 meV/A in those 60 epochs, and 4.2 to 4.3 once its averaging
  constants were tuned. A 30-epoch SOAP run (4.76) now beats the earlier 120-epoch recipes.
- With SOAP the training error is below the validation error (3.4 against 4.5 meV/A at 40 epochs), so the 300
  training frames now set the limit more than the optimiser does.

## What the numbers refer to

- **Task:** distilling MACE-MH-1 (head `omat_pbe`) into the `s64l0` student on SrTiO3: 300 training frames, 30
  validation frames, 60 teacher-test frames.
- **Metric:** validation force RMSE in meV/A of the EMA model ("valid F"). Lists such as 5.73 / 5.76 / 5.82 are
  seeds.
- **Hardware:** RTX 5060 Ti, with further seeds on RTX 4000 Ada workstations.
- **Comparability:** the recipes changed while these runs were made (batch size, schedule, sampling, loss
  weights). Compare figures within one row or one section; across sections only the direction is safe.

## How the three are used here

All three run inside the training library's fused step, so the optimiser adds kernels to the step and no Python.

| Optimiser | Applied to | Update |
| --- | --- | --- |
| Adam (AMSGrad) | every parameter in the baseline; in the Muon and SOAP recipes only the species embedding, the readouts and the small contraction weights | per-element first and second moments |
| Muon | 27 matrix blocks: the radial MLP layers, each degree's block of the e3nn linear maps, `skip_tp` and `weights_max` per species | Nesterov momentum, then the momentum matrix is made orthogonal by Newton-Schulz steps (8 + 2 in the recipes), scaled by `0.2 sqrt(max(m, n))` |
| SOAP | the same 27 blocks | running averages of `G G^T` and `G^T G` per matrix; their eigenbases are refreshed every 10 steps; Adam runs in that rotated basis |

Two additions to SOAP are ours:

- **Blocked bases (`soap_block`):** one eigenbasis per e3nn degree block of a matrix, not one for the whole
  matrix. The blocks never mix under the model's symmetry, and the smaller eigenproblems converge in fewer
  sweeps.
- **Cautious Adam (`cautious: adam`):** on the tensors that stay with Adam, an update is masked where its sign
  disagrees with the gradient.

## Adam

Adam was the starting point because it is what mace trains with.

| Run (batch 4) | Valid F | Notes |
| --- | --- | --- |
| mace's own 150-epoch recipe | 12.8 | test 9.0; about 40 minutes in stock mace on an L4 |
| our 60-epoch schedule, gradient clip 1, peak lr 0.05 | 12.0 to 12.9 | 39 to 40 s of steps and validation |
| the same without mace's L2 weight decay | 11.0 / 11.45 | |
| 240 epochs | 8.14 | still falling at the end |

What we learned:

- **The schedule mattered a great deal and still left Adam slow.** Lower learning rates learned more slowly;
  rates of 0.02 and above gave spikes in the force error until the gradient norm was clipped at 1 (mace clips at
  10). Clipping was the largest single lever found for Adam (13.2 to 10.8 at 100 epochs).
- **Adam was not converged at 60 epochs.** The error kept falling to 240 epochs, and the gap between training and
  validation error was small. We read that as an optimiser limit, which is why we tried others.
- **mace's L2 weight decay (5e-7) cost about 1 meV/A.** It acts at the full learning rate on parameters whose
  gradients are small.
- **Larger batches hurt.** Batch 8 gave 14.45 against 12.22 at batch 4.
- **Small changes did nothing.** AMSGrad off, other betas, a StableAdamW-style update clip and weight ramps all
  tied or lost.

## Muon

Muon was tried first as the replacement for the matrix parameters.

| Run | Valid F | Notes |
| --- | --- | --- |
| batch 4, 60 epochs, first library version | 8.0 to 8.15 | Adam in the same setting: 12.0 to 12.9 |
| with 8 + 2 Newton-Schulz steps and no weight decay (`fast60`) | 7.25 to 7.39 | |
| 240 epochs | 5.86 | test 3.6; 161 s wall |
| batch 1, 60 epochs | 5.76 / 5.81 | |
| batch 4, 60 epochs, momentum 0.99 (found later) | 5.82 / 5.92 / 5.92 | from 7.29 / 7.29 / 7.38 at 0.95 |

What we learned:

- **A large gain over Adam at equal epochs,** and it passed 13 meV/A (Adam's 60-epoch level) by epoch 19.
- **L2 weight decay must not reach the Muon tensors.** Muon's step does not shrink with the gradient, so late in
  training the decay term became a sizeable part of the orthogonalised direction: 9.56 with it, 7.62 without.
  The library now leaves it out for those tensors.
- **Momentum is the sensitive setting.** Raising it from 0.95 to 0.99 gave more than 1 meV/A at 60 and 120
  epochs, but 0.995 diverged, and 0.99 diverged in the 30-epoch recipe with its higher peak learning rate.
- **Of DPA4's HybridMuon additions, only the extra Newton-Schulz steps helped** (7.42 against 7.65 in the torch
  reference). Its Magma-lite damping (8.22), decoupled weight decay 1e-3 (7.88) and its learning-rate schedule
  (7.75) did not.
- **Variants of the direction lost or tied.** A per-block RMS normalisation in place of the orthogonal step gave
  13.89, which shows the orthogonalisation itself is what helps. NorMuon tied (7.22 to 7.29).
- **Fewer, larger steps lose.** Batch 2 beat batch 4 at a fixed epoch count (6.31 against 7.28), which led to the
  batch-1 recipes.

## Why SOAP

SOAP was first run as one more screen on the Muon recipe and won clearly, so it became the default for the named
`soapNN` recipes.

At 60 epochs, with the settings of the time (before the beta tuning below):

| Setting | Muon | SOAP | SOAP, cautious Adam, 2 sweeps | SOAP, blocked bases as well |
| --- | --- | --- | --- | --- |
| batch 4 | 7.25 to 7.39 | 6.08 / 6.08 / 6.16 | 5.77 to 5.91 | 5.73 / 5.76 / 5.82 |
| batch 1 | 5.76 / 5.81 | 5.80 / 5.94 | 5.58 / 5.62 | 5.48 / 5.57 |

At batch 1 against batch-1 Muon by budget: 30 epochs 6.20 to 6.33 against 6.78, 40 epochs 5.94 to 5.97 against
6.31, 60 epochs 5.42 to 5.59 against 5.76. SOAP saved about a third of the epochs at 30 to 40.

The reasons we stayed with it:

1. **Better error at equal epochs and at equal time.** `soap40` reaches 4.52 in 23 s of steps; the best batch-4
   Muon recipe (`fast60`, momentum 0.99) reaches 5.89 in 31.5 s.
2. **It costs less per step than Muon here.** At batch 1: Adam 1.83 ms, Muon with 8 + 2 Newton-Schulz steps 2.17
   ms, blocked SOAP refreshed every 10 steps 2.03 ms, and 1.94 to 1.95 ms after the SOAP kernels were reworked.
   Muon pays for its Newton-Schulz iteration on every step; SOAP pays for an eigenbasis refresh once in ten.
3. **It had a large reserve in its averaging constants.** See the next section: 0.95 to 0.995 took a further 1
   meV/A off at every budget.
4. **Its schedule is forgiving.** Once the betas were set, peak learning rate, warm-up, decay shape, EMA, clipping
   and the refresh interval all tied within the seed spread over wide ranges.

One caution on the comparison: Muon's better momentum (0.99) was found after SOAP's betas, and we have no
batch-1 Muon run at that momentum. The gap between a fully tuned batch-1 Muon and SOAP is therefore not
measured. The batch-4 comparison in point 1 is the closest one we have.

## SOAP settings that mattered

**The betas.** SOAP's published values (0.95 / 0.95) come from large-batch language-model training. Here, at
batch 1 with about 300 steps per epoch, much longer averages are better:

| Recipe | 0.95 / 0.95 | 0.995 / 0.995 | 0.997 / 0.997 |
| --- | --- | --- | --- |
| `soap30` | 5.80 | 4.76 | 4.80 |
| `soap40` | 5.50 to 5.56 | 4.52 | 4.50 |
| `soap60` | 5.22 | 4.28 | 4.21 |
| `soap120` | 4.88 | 4.00 | 3.89 |

- Both betas have to rise together. Beta1 at 0.99 or above with beta2 at 0.95 diverges.
- 0.999 loses again (5.10 at 40 epochs).
- The 30- and 40-epoch recipes use 0.995; the 60- and 120-epoch ones use 0.997.
- With beta1 equal to beta2 the step is an average over about 200 steps (two thirds of an epoch) of the rotated
  gradient over its RMS.

**Other settings that moved the result:**

- Learning-rate multiplier 1.5 for the SOAP blocks against the Adam tensors (2.0 and 2.5 were worse before the
  beta change and tied after it).
- Beta2 0.99 for the tensors that stay with Adam.
- Batch 1. Batch 2 with sampling ties it at equal time; batch 4 loses.

**Settings that tied or lost** (so they stay at the simple value): refresh every 5, 20 or 40 steps instead of
10; Nesterov off; decoupled weight decay on the SOAP blocks (1e-4 to 1e-3 tie, 3e-3 loses); EMA 0.98 to 0.998;
gradient clip 3; cosine and geometric decays (both worse than linear).

## Known limits

- **SOAP runs are not bit-reproducible.** Two runs of one setting differ by 0.1 to 0.2 in the parameters after 30
  steps, even in FP64. The order of atomic adds in the gradients differs at 1e-16, and the eigenbasis rotations of
  near-degenerate eigenvalues amplify it. Plain Muon stays at 1e-14. The seeds land within the usual spread, so
  compare recipes over several seeds.
- **Matrix sides above 64 are left unrotated** (`soap_max_dim`).
- **One system.** Everything above is SrTiO3 with 300 frames and the 64-channel scalar student. The betas and the
  batch size depend on the number of steps per epoch and on the gradient noise, so they should be screened again
  on other data sets and widths.
- **Data limit.** At `soap40` the errors are 3.4 (train), 4.5 (validation) and 2.8 (teacher test) meV/A; at
  `soap120` 2.5 / 3.8 / 2.4. A further gain at a fixed budget is more likely to come from more or other training
  frames than from the optimiser.

## Where it stands now

The standalone trainer `loupstrain_cli` keeps SOAP (a BF16 tensor-core version with a QR-based refresh) and
trains this student in 21 epochs: 2.6 s of GPU time, valid F 4.37 to 4.49, teacher-test F 2.74 to 2.82 meV/A.
Stock mace with Adam and AMSGrad on the same 300 frames took about 7,230 s (L4, FP64) for a test F of 8.54 to
9.23. That comparison includes the kernel work, per-atom teacher energies, a different cutoff and a closed-form
readout solve as well as the optimiser, so it is not a measure of SOAP alone. In that trainer, full-batch L-BFGS,
SOAP betas below 0.995 and batch 2 were tried and rejected.

## References

- Muon: Jordan et al. 2024.
- SOAP: Vyas et al. 2024.
- Cautious updates: Liang et al. 2024.
- HybridMuon: DPA4, arXiv 2606.02419, Supplementary S-2.
