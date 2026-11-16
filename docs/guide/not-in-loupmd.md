# What LoupMD leaves out

Three FastMACE features are not in LoupMD on purpose (PLAN.md decision A3). Each FastMACE spelling raises
`PlanRefused`, and the message names what to use instead.

| FastMACE | In LoupMD | Use instead |
|---|---|---|
| `precision="float64"` (float64 forces) | `lm.load(..., precision="float64")` is refused | float32, the default. Every supported model passes the 3x float32-vs-float64 gate (`loupmd validate --model LABEL`). For float64 reference values, run MACE itself with `default_dtype="float64"`. |
| `allow_reference=True`, `fallback=` (the MACE calculator as a fallback) | refused by `lm.md` | Run the model with `mace.calculators.MACECalculator` and ASE's MD drivers. A model LoupMD cannot run is refused with the reason (`loupmd models` lists what it runs). |
| `Workload(reproducible=True)` (bitwise-reproducible runs) | `lm.config.Workload(reproducible=True)` is refused | `reproducible=False`. The seed fixes the thermostat's noise, and `Simulation.resume` continues a run from its saved state. Float32 atomic sums on the GPU differ in their last bits from run to run. |

FastMACE's other execution switches (`mode=`, `runner=`, `engine=`, `neighbours=`, `calibrate=`) select its
internals. LoupMD has one execution path, so `lm.md` refuses them too.
