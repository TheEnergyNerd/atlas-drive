> This repository has moved to [pranavatlas/atlas-drive](https://github.com/pranavatlas/atlas-drive). Use that repository for current code and issues.

# Atlas Drive

**Design the motor and motion together.**

Atlas designs and manufactures motors and actuators. This repository is the open-source runtime for using actuator envelopes in robot simulation: a portable model format, NumPy and PyTorch evaluators, and an experimental Isaac Lab adapter.

[Research and interactive examples](https://pranavatlas.github.io/atlas-drive-research/) · [Have Atlas design and manufacture your actuator](https://atlasmotion.com/contact)

## Start locally

```sh
git clone https://github.com/pranavatlas/atlas-drive.git
cd atlas-drive
python -m pip install .
atlas-drive examples/synthetic-envelope.json --torque 25 --speed 10 --temperature 25
```

This synthetic example delivers **20 Nm** against a **25 Nm** request at **10 rad/s**. The result reports saturation. All quantities refer to the output shaft. No account, cloud connection or uploaded workload is required.

```python
from atlas_drive_runtime import EnvelopeProfile, evaluate

profile = EnvelopeProfile.load('examples/synthetic-envelope.json')
result = evaluate(profile, requested_torque_Nm=[25, -25], speed_rad_s=[10, -10])
print(result['torque_Nm'])  # [20, -20]
print(result['saturated'])  # [True, True]
```

## Use it in robot learning

```sh
python -m pip install '.[torch]'
```

```python
from atlas_drive_runtime.torch_backend import TorchEnvelope, TorqueState
import torch

envelope = TorchEnvelope(profile, device='cpu')  # or 'cuda'
drive = TorqueState(envelope, shape=(128, 6))
request = torch.full((128, 6), 25., device=envelope.device)
speed = torch.full_like(request, 10.)
result = drive.step(request, speed, temperature=25., dt=0.0025)
# Apply result['torque_Nm']; handle result['fault'] in your environment.
# Reset only the environments whose episodes ended:
drive.reset(torch.tensor([0, 7], device=envelope.device))
```

The PyTorch path evaluates the whole batch on the selected device. It supports signed torque limits, speed/temperature interpolation, optional response lag, and independent environment resets. The NumPy evaluator is a stateless reference for parity checks.

The [Isaac Lab integration guide](docs/isaaclab.md) shows how to replace a joint's explicit actuator with the supplied envelope adapter. The adapter targets **Isaac Lab 2.1.1**. CPU runtime tests are included; the adapter is experimental until the integration smoke test is run in your Isaac installation. This release does not claim new robot training or hardware results.

## What a model contains

A profile supplies lower and upper output-torque limits over a grid of signed speed and externally supplied temperature. [The format](docs/profile-format.md) specifies units, interpolation and out-of-range behavior. The optional response time is a first-order torque response, not an inverter simulation.

Use your own characterized envelope or request a model from Atlas. The example in this repository is **entirely synthetic**. It contains no Atlas bench records, measured calibration, motor dimensions, winding recipes, supplier information or manufacturing costs.

A profile is an exported behavior model. Atlas's electrical design search, bench/calibration data and manufacturing workflows are not part of this repository. A profile can be generated from a more detailed model, but this runtime does not reproduce the private solver or independently establish the profile's accuracy.

Temperature is an input, not a prediction. Mass, rotor inertia, friction, contact and robot geometry remain in the robot model. Finite grids approximate the supplied limits between samples; validate their resolution against the source model. An out-of-range or nonfinite state produces zero commanded torque and a fault flag. This is a simulation policy, not an electrical safety guarantee or a prediction of passive braking.

## From the job to the actuator

Bring Atlas the torque/speed workload, motion, packaging constraints and duty cycle. Atlas can propose the motor and drive requirements, manufacture the actuator and supply a model for your controls team. [Contact Atlas](https://atlasmotion.com/contact).

## Development

```sh
python -m pip install -e '.[test,torch]'
python -m pytest -q
```

Tests cover signed limits, interpolation, thermal-input derating, out-of-range faults, NumPy/PyTorch parity, response lag and selective reset. GPU parity runs automatically when CUDA is available; otherwise it is skipped. See [CONTRIBUTING.md](CONTRIBUTING.md) before attaching robot logs or models.

MIT licensed. Isaac Lab, PyTorch and NumPy are separate dependencies under their own licenses.
