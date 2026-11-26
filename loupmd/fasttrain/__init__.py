"""Loup-S training engine: a fast, exact FP32 trainer for stock two-layer scalar-sender MACE students.

The engine replaces mace-torch's energy/derivative evaluation and per-step data path while keeping
mace-torch's model object as the canonical parameter store. Optimizer, EMA, SWA, checkpoints and the
exported ``student.model`` therefore stay mace-torch's own objects (plans/loup-s-training-engine.md).
"""
