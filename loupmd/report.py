"""The one-screen report of a run: header, physics, plan, gate, setup, speed, licence, warnings.

Made from the run's metadata (``loupmd.provenance``), so the screen and ``metadata.json`` cannot disagree.
"""
from __future__ import annotations

from . import models


def _bytes(n) -> str:
    return f"{n / 2 ** 30:.2f} GiB"


def _model_line(model) -> str:
    name = model["label"] or model["file"]
    head = f", head {model['head']}" if model["head"] else ""
    return f"{name} ({model['file']}, SHA-256 {model['sha256'][:16]}...{head})"


def _structure_line(structure) -> str:
    elements = ", ".join(f"{symbol} {count}" for symbol, count in structure["elements"].items())
    lengths = " x ".join(f"{sum(x * x for x in row) ** 0.5:.2f}" for row in structure["cell"])
    source = structure["source"] or "Atoms"
    repeat = structure["repeat"]
    if repeat and tuple(repeat) != (1, 1, 1):
        source += " x ({}, {}, {})".format(*repeat)
    fixed = f", {structure['fixed_atoms']} fixed" if structure.get("fixed_atoms") else ""
    return f"{source}: {structure['atoms']} atoms ({elements}{fixed}), cell {lengths} A"


def _physics_lines(metadata) -> list:
    integrator, schedule = metadata["integrator"], metadata["schedule"]
    timestep = integrator["timestep_fs"]
    if integrator["scheme"] == "nve":
        first = f"NVE, velocity Verlet, timestep {timestep:g} fs"
        if integrator.get("fixcm"):
            first += ", total momentum removed"
    else:
        first = (f"NVT, Langevin, timestep {timestep:g} fs, friction {integrator['friction_per_fs']:.4g}/fs")
        if schedule is None:
            first += f", {integrator['temperature_K']:g} K"
        if integrator.get("fixcm"):
            first += ", centre of mass at rest"
    lines = [first]
    if schedule is not None:
        parts = [f"{s['phase']} {s['start_K']:g} -> {s['end_K']:g} K in {s['ps']:g} ps ({s['steps']} steps)"
                 if s["kind"] == "ramp" else f"hold {s['start_K']:g} K for {s['ps']:g} ps ({s['steps']} steps)"
                 for s in schedule["segments"]]
        lines.append("schedule: " + "; ".join(parts))
    steps = metadata["counters"]["steps"]
    lines.append(f"{steps} steps ({steps * timestep / 1000:g} ps) to step {metadata['step']}, seed {metadata['seed']}")
    return lines


def _plan_lines(metadata) -> list:
    plan, gpu = metadata["plan"], metadata["gpu"]
    where = (f"GPU {gpu['index']} ({gpu['name']}, compute capability {gpu['compute_capability']}, driver "
             f"{gpu['driver']})") if gpu else "the GPU"
    lines = [f"the whole system on {where}"]
    if plan["kernels"]:
        layers = f", {plan['interaction_layers']} interaction layers" if plan["interaction_layers"] else ""
        cutoff = f", cutoff {plan['r_max_A']:g} A" if plan["r_max_A"] else ""
        lines.append(f"{plan['kernels']} kernels{layers}{cutoff}; library {plan['library'] or 'unknown'}")
    if plan["memory"]:
        lines.append(f"memory: {_bytes(plan['memory']['whole_bytes'])} planned, {_bytes(plan['memory']['free_bytes'])} "
                     "free on the card")
    lines.append(plan["precision"])
    return lines


def _gate_line(gate) -> str:
    status = gate["status"]
    if status == "skipped":
        return 'skipped (check="skip"): run `loupmd validate` or pass check="gate" to compare with MACE float64'
    if status == "unavailable":
        return f"unavailable: {gate.get('why', 'the references could not be computed')}"
    worst = gate["worst"]
    return (f"{status}: worst ratios energy {worst['energy_ratio']:.2f}, force {worst['force_ratio']:.2f}, stress "
            f"{worst['stress_ratio']:.2f} of the 3 allowed, on {len({row['config'] for row in gate['rows']})} "
            "rattled copies")


def _speed_line(metadata) -> str:
    timing, counters = metadata["timing"], metadata["counters"]
    if not timing["ms_per_step"]:
        return "no step ran"
    return (f"{timing['ms_per_step']:.3f} ms/step with output ({timing['engine_ms_per_step']:.3f} in the engine), "
            f"{timing['ns_per_day']:.1f} ns/day, {timing['atom_steps_per_s'] / 1e6:.2f} million atom-steps/s; "
            f"{counters['rebuilds']} neighbour rebuilds, {counters['dangerous_builds']} dangerous")


def _licence_line(model) -> str:
    text = models.LICENCE_TEXT.get(model["licence"])
    return (f"model: {text}" if text else "model: not in the label table, see its publisher") + "; LoupMD: MIT"


def _block(title, lines) -> list:
    return [title] + ["  " + line for line in lines]


def text(metadata) -> str:
    """The report of a run from its metadata."""
    out = [f"LoupMD {metadata['loupmd']['version']}: MD run, status {metadata['status']}"]
    out += _block("Model and structure", [_model_line(metadata["model"]), _structure_line(metadata["structure"])])
    out += _block("Physics", _physics_lines(metadata))
    out += _block("Plan", _plan_lines(metadata))
    out += _block("Gate", [_gate_line(metadata["gate"])])
    setup = metadata["timing"]["setup_s"]
    bundle = metadata["bundle"] or "the bundle"
    out += _block("Setup", [f"{bundle}; {setup:.1f} s for export or cache, engine and the first force evaluation"
                            if setup is not None else "not started"])
    out += _block("Speed", [_speed_line(metadata)])
    out += _block("Licence", [_licence_line(metadata["model"])])
    if metadata["warnings"]:
        out += _block("Warnings", list(metadata["warnings"]))
    if metadata["output"]:
        out += _block("Output", [f"{metadata['output']['directory']}: thermo.*.csv every "
                                 f"{metadata['output']['thermo_every']} steps"
                                 + (f", trajectory.*.traj every {metadata['output']['trajectory_every']}"
                                    if metadata["output"]["trajectory_every"] else "")
                                 + ", restart.npz, simulation.json, metadata.json"])
    return "\n".join(out)
