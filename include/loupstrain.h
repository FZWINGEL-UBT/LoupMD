/* Loup-S training library (libloupstrain.so): optimizer steps of the Loup-S student (mace-torch 0.3.16
 * ScaleShiftMACE, scalar hidden features, two interactions, l <= 3, correlation 3) on the GPU.
 *
 * One step = gather a padded batch, energy, forces and stress (first reverse sweep), the weighted
 * energy / forces / stress loss, the parameter gradient (second reverse sweep), global-norm clipping,
 * Adam / AMSGrad and the EMA update, all captured in one CUDA graph and replayed per step.
 *
 * The caller (loupmd/fasttrain/cuda.py) owns every tensor: mace's parameters, Adam state and EMA shadows,
 * the device-resident frames and each epoch's stacked batch indices are given to the library as device
 * pointers and updated in place. Pointers must stay valid (and at the same address) until the engine is
 * destroyed or the pointer is set again.
 */
#ifndef LOUPSTRAIN_H
#define LOUPSTRAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LST_ABI_VERSION 6
#define LST_N_PARAMS 26 /* mace named_parameters() order, see lst_set_params */

typedef struct LstEngine LstEngine;

typedef struct LstModelDesc {
    int dtype;      /* 0 float32, 1 float64 */
    int S;          /* species */
    int C;          /* scalar channels (64) */
    int H;          /* radial MLP hidden width (64) */
    int NB;         /* radial basis functions (8) */
    int hid;        /* readout hidden width (16) */
    int K;          /* product polynomials (28, must equal the generated POLY_K) */
    int K3, K2;     /* of which cubic (weights_max) and quadratic (weights.0); the rest linear (weights.1) */
    int p_cut;      /* polynomial cutoff exponent */
    double r_max, prefactor, act, scale, avg0, avg1;
    const double* bessel_w; /* [NB], host */
} LstModelDesc;

typedef struct LstShape {
    int N;  /* padded nodes per batch */
    int E;  /* padded directed edges (even) */
    int P;  /* edge pairs = E / 2 */
    int G1; /* graph slots including the dummy graph */
    int B;  /* graphs per full batch (G1 - 1) */
} LstShape;

/* Device arrays of every frame, rows concatenated, with one appended dummy row (dtype of the engine unless
 * noted). Per-frame arrays have one row per frame plus the dummy frame. */
typedef struct LstFrames {
    const void* vectors;  /* [edges + 1, 3] */
    const int64_t* species; /* [nodes + 1] */
    const void* f_ref;    /* [nodes + 1, 3] */
    const void* core_f;   /* [nodes + 1, 3] */
    const void* e_ref;    /* [frames + 1] */
    const void* core_e;
    const void* volume;
    const void* n_atoms;
    const void* w_e;
    const void* w_f;
    const void* w_s;
    const void* s_ref;    /* [frames + 1, 9] */
    const void* core_s;   /* [frames + 1, 9] */
    /* topology of every frame (int32, frame-local indices; edges receiver-sorted, every edge with its reverse) */
    int n_frames;                 /* the dummy frame / node row / edge row is the one after the last */
    int dummy_row;                /* slack edges per dummy node of a padded batch */
    const int32_t* node_off;      /* [frames + 1] first node row of each frame */
    const int32_t* edge_off;      /* [frames + 1] first edge row */
    const int32_t* pair_off;      /* [frames + 1] first pair */
    const int32_t* sender;        /* [edges] */
    const int32_t* receiver;      /* [edges] */
    const int32_t* receiver_ptr;  /* [nodes + frames]: frame f's n_f + 1 entries start at node_off[f] + f */
    const int32_t* pidx;          /* [edges] pair of each edge */
    const int32_t* pe0;           /* [pairs] first edge of each pair */
    const int32_t* pe1;           /* [pairs] its reverse */
} LstFrames;

/* One epoch's batches: the frames of each (host array, copied by lst_begin_epoch). Each step assembles its
 * padded batch on the device: the frames' nodes and edges in order, then dummy nodes, slack self-loops on them
 * (dummy_row per node) and the dummy graph. The caller guarantees that every batch fits the shape. */
typedef struct LstBatches {
    int n;                       /* steps */
    const int32_t* frame_ids;    /* [n, B] frame of each graph slot, -1 after the batch's last frame */
} LstBatches;

/* Adam / AMSGrad state of the 26 parameters (torch.optim.Adam's state tensors), and EMA shadows. */
typedef struct LstOptim {
    void* exp_avg[LST_N_PARAMS];
    void* exp_avg_sq[LST_N_PARAMS];
    void* max_exp_avg_sq[LST_N_PARAMS]; /* NULL when amsgrad is off */
    float* step[LST_N_PARAMS];          /* float32 device scalars */
    void* ema[LST_N_PARAMS];            /* NULL: no EMA */
    int group[LST_N_PARAMS];            /* parameter group of each parameter */
    int n_groups;
    double beta1, beta2, eps;
    double weight_decay[16];            /* per group */
    double max_grad_norm;               /* <= 0: no clipping */
    double update_clip;                 /* > 0: per tensor, lr / max(1, RMS(g / sqrt(v_hat)) / update_clip)
                                           (StableAdamW); <= 0: off */
} LstOptim;

int lst_abi_version(void);
const char* lst_last_error(const LstEngine* h);

LstEngine* lst_create(const LstModelDesc* model, const LstShape* shape, int device);
void lst_destroy(LstEngine* h);

/* mace's 26 trainable tensors (device pointers, engine dtype), in model.named_parameters() order:
 * node_embedding.linear, interactions.0.{linear_up, conv_tp_weights.layer0..3, linear, skip_tp},
 * interactions.1.{same}, products.0.{weights_max, weights.0, weights.1, linear}, products.1.{same},
 * readouts.0.linear, readouts.1.{linear_1, linear_2}. */
int lst_set_params(LstEngine* h, void* const* params);
int lst_set_frames(LstEngine* h, const LstFrames* frames);
/* The order in which the force pass takes each frame's nodes: device int32 [nodes], frame-local node indices
 * in the frames' node layout, a permutation of each frame (NULL: index order). The frames' own atom order (the
 * inverse of the species sort) keeps both edges of a pair close in time, so the pair's radial rows are read
 * from L2 the second time. Call after lst_set_frames (which clears it); the array must outlive the engine. */
int lst_set_node_order(LstEngine* h, const int32_t* order);
/* Gauge-centered per-atom teacher local energy distillation (Wave 30): ei_ref is device [nodes] (engine dtype)
 * containing target MLIP per-atom local energies (REF_node_energy - core_ei) in frames' node layout (NULL or
 * weight <= 0 disables). Loss = (w_ei / n_real) sum_n ((ne[n] - ei_ref[n]) - mean_{g,s}(ne - ei_ref))^2,
 * invariant to per-species E0/shift gauge offsets and orthogonal to frame total energy. Call after lst_set_frames.
 * Optional symbols (older libraries lack them). */
int lst_set_node_energy(LstEngine* h, const void* ei_ref, double weight);
int lst_set_node_energy_steps(LstEngine* h, const double* w_ei_steps, int n_steps, void* stream);
int lst_set_optim(LstEngine* h, const LstOptim* optim);

/* Muon (ABI 5): matrices of the trainable tensors updated with orthogonalised momentum instead of Adam.
 * A block is `batch` matrices of rows x cols inside tensor `param`; element (b, r, c) is at
 * offset + b * batch_stride + r * row_stride + c * col_stride of the tensor. Per step, for every tensor holding
 * a block: g (clipped; no L2 weight decay since ABI 6), exp_avg = momentum * exp_avg + g, the direction
 * D = g + momentum * exp_avg (nesterov) or exp_avg; per matrix O = NS5(D / |D|_F) (quintic Newton-Schulz,
 * coefficients 3.4445, -4.7750, 2.0315) and p -= lr * scale * sqrt(max(rows, cols)) * O; then its EMA. Elements
 * of such a tensor outside every block are left unchanged. n = 0 turns Muon off (every tensor Adam). */
typedef struct LstMuonBlock {
    int param, batch, rows, cols;
    long offset, batch_stride, row_stride, col_stride;
} LstMuonBlock;
int lst_set_muon(LstEngine* h, const LstMuonBlock* blocks, int n, double momentum, int nesterov, double scale);

/* Muon's Newton-Schulz iteration (ABI 6): `steps` quintic steps (default 5), then `polish` steps with
 * (2, -1.5, 0.5) (default 0; DPA4 uses 8 + 2), steps + polish <= 32. */
int lst_set_muon_ns(LstEngine* h, int steps, int polish);

/* NorMuon (2510.05491): after Newton-Schulz each block row is divided by the root of an EMA (beta) of its mean
 * square and the block rescaled to the orthogonalised update's Frobenius norm. beta < 0 turns it off (default);
 * a call resets the rows' state. Optional symbol (older libraries lack it). */
int lst_set_muon_norm(LstEngine* h, double beta);

/* Cautious Adam (muon.py cautious='adam'): on the tensors Adam updates, only the components whose step opposes
 * the gradient move, and the tensor's step is scaled by numel / (kept + 1). on = 0 turns it off (default).
 * Optional symbol (older libraries lack it). */
int lst_set_cautious(LstEngine* h, int on);

/* Decoupled weight decay of the Muon / SOAP blocks (muon.py muon_wd): before each step their parameters are
 * scaled by 1 - lr * wd_mult, lr being the block's group rate and wd_mult = muon_lr_mult * muon_wd. 0 turns it off
 * (default). Optional symbol (older libraries lack it). */
int lst_set_muon_wd(LstEngine* h, double wd_mult);

/* Arithmetic from the next step on: gemm = the FP32 GEMMs' cuBLAS compute type, mlp = the radial MLPs' kernels;
 * 0 FP32, 1 TF32, 2 bf16, -1 keeps that part (LST_MATH / LST_MLP_TC set them at creation; the MLPs' tensor-core
 * modes need them set up there). The step graph is captured again. FP64 engines ignore it. Optional symbol. */
int lst_set_math(LstEngine* h, int gemm, int mlp);

/* SOAP (Vyas et al. 2024) on the Muon blocks instead of Newton-Schulz (after lst_set_muon; every <= 0 turns it
 * off; lst_set_muon resets it). Adam (b1, b2, eps; momentum (1 - b1) g in exp_avg) in the eigenbasis of the EMA
 * (b2) factors X X^T and X^T X, refreshed every `every` steps by cyclic Jacobi from the previous basis. The
 * smaller side (<= 64) is always preconditioned, the larger one only up to max_dim (<= 64), else identity.
 * Step: lr of the group times lr_mult. Optional symbol. */
int lst_set_soap(LstEngine* h, double b1, double b2, double eps, int every, int max_dim, double lr_mult);

/* Before an epoch: its batches, the learning rate of each group, the loss weights (energy, forces, stress)
 * and the EMA decay of each step (host array of n, ignored without EMA). Resets the step index to 0. */
int lst_begin_epoch(LstEngine* h, const LstBatches* batches, const double* lr, const double* loss_weights,
                    const double* ema_decay, void* stream);

/* As lst_begin_epoch with a schedule per step: lr_steps [n, n_groups] (each step's learning rate of every
 * group), loss_weights_steps [n, 3] (energy, forces, stress of each step); ema_decay as above. */
int lst_begin_epoch_steps(LstEngine* h, const LstBatches* batches, const double* lr_steps,
                          const double* loss_weights_steps, const double* ema_decay, void* stream);

/* Run the next n optimizer steps of the epoch on the stream (the first call captures the graph). */
int lst_run_steps(LstEngine* h, int n, void* stream);

/* Losses of the steps run so far in this epoch (host array of at least the epoch's n); synchronises. */
int lst_epoch_losses(LstEngine* h, double* out, void* stream);

/* Energies [G1], forces [N, 3] (rows in batch order: each frame's atoms after the previous frames') and stress
 * [G1, 9] of frames frame_ids[0..n) (1 <= n <= B, fitting the padded shape) under the current parameters, into
 * the caller's device buffers in the engine dtype: the step's forward and force pass on the stream, no loss,
 * backward or update, no graph. Its own frame ids and step: the epoch set by lst_begin_epoch is untouched. */
int lst_evaluate(LstEngine* h, const int* frame_ids, int n, void* energy, void* forces, void* stress, void* stream);
/* Frozen-weights eval cache (Wave 32): when on=1, repeated lst_evaluate calls skip redundant view/MLP recomputation
 * and replay a captured CUDA graph; call with on=0 before mutating params in-place! Optional symbol. */
int lst_set_eval_cache(LstEngine* h, int on);

/* Debugging: one step without the optimizer update (graph not used) and copies of named stages. */
int lst_debug_step(LstEngine* h, int batch, int update, void* stream);
long lst_stage_size(LstEngine* h, const char* name);           /* elements, -1 if unknown */
int lst_stage_copy(LstEngine* h, const char* name, void* dst, void* stream); /* device-to-device */
const void* lst_stage_ptr(LstEngine* h, const char* name);     /* internal device pointer (NULL if unknown) */

#ifdef __cplusplus
}
#endif
#endif
