# Critical review of the LoupMD manuscript

The Methods and Related Work sections now describe a fixed pretrained checkpoint from energy evaluation through force calculation, tiling and integration. The audit used LoupMD `release/v1` at `430c4aa4256c801b5a2a44dfdea713fcf31b4de7`.

| Finding in the original draft | Change in the manuscript |
| --- | --- |
| The validation equation left the impression that both POLAR stress errors use the same reference. | §1.7 gives both errors explicitly. The development test uses the native float64 energy derivative under strain for the candidate stress. The tolerance uses the ordinary float32 and float64 calculator stresses. The text also gives the packaged validator's finite-difference target, the three required structures and the default POLAR-1-S exclusion. |
| Several optimisation citations began with recent MACE evaluators even though an earlier native implementation already described the technique. | §§1.2–1.3 cite Kovács et al. at cubic radial tables, fused neighbour sums, sparse products and corrected tensor cores. §2 distinguishes the original Symmetrix and `cuda_mace` implementations from Symmetrix-XL. The cubic Hermite construction, reverse edge recomputation, three-sweep schedule and captured MACE chunks each have a local reference. |
| The eight-byte table description merged two different layers and two different error checks. | §1.2 explains the first-layer species reduction with unpacked coefficients, the later packed convolution tables, and the separate density column. The spline fit is tested against the native radial network. The packing check bounds value and derivative errors relative to the unpacked cubic. |
| The draft could be read as accumulating the GEMM correction in FP16 and retrying a failed trajectory. | §1.3 specifies the two cross products, FP16 inputs, FP32 accumulation and a sufficient input bound. It limits automatic retry to a standalone whole-cell evaluation and identifies the restart requirement for an affected MD block. |
| Plain tiling appeared to replay the first layer unconditionally, and invalid neighbour-row order appeared recoverable by rebuilding. | §§1.4–1.5 explain when plain tiling retains enough state to skip replay, when the two-layer wavefront releases a row, and when the host handles a capacity overflow. An ordering error suppresses the force pass and terminates the calculation. |
| Throughput comparisons did not say exactly what LoupMD timed. | §1.5 specifies the moving-atom Langevin protocol and the block clock. §2 identifies the static-coordinate run with zero rebuilds in the matched Symmetrix-XL comparison and the different operations timed for MACE and DPA4C in DPA4C Table 1. |

## What the evidence supports

The example of 119,833 live rows among 1,081,600 atoms describes first-layer storage in one recorded OMAT-0 medium run on an RTX 5060 Ti. Geometry and tile order determine the fraction. A cross-engine speedup would need measurements with the same checkpoint, precision, hardware and timed operation. The manuscript review used the existing run record and release source. [Profiling record](https://github.com/FZWINGEL-UBT/LoupMD/blob/430c4aa4256c801b5a2a44dfdea713fcf31b4de7/STATUS_AND_PROFILE.md#L1140) · [Claim and input evidence](evidence.md)

## Citations and verification

The bibliography has 56 entries. The citation map records their locations and source passages. Two primary sources were added during review. Nigam et al., Eq. 4, credits the recursive construction of equivariant features. Baldwin's MACE pull request credits the reciprocal-space electrostatic stress correction. Their entries in the citation map identify this provenance. [Nigam et al.](https://doi.org/10.1063/5.0021116) · [MACE stress correction](https://github.com/ACEsuit/mace/pull/1718/commits/26d27bd62e9669731a2e743c873a8291669a3bb9) · [Citation map](citation_map.json)

The original and revised sections went through Paper Orchestra's peer-review rubric, with three independent readers per round and two revisions. The release-code claims were checked against the pinned source revision. The standalone LaTeX article was rebuilt, its citation destinations were checked and all eight pages were inspected.
