# Isaac Lab integration

The adapter implements Isaac Lab's explicit actuator interface and targets version 2.1.1. Import it **after** `AppLauncher` initializes Isaac Sim. The interface follows the upstream [ActuatorBase](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/actuators/actuator_base.py) and [actuator configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/actuators/actuator_cfg.py) contracts.

Install this checkout using the Python environment already used by Isaac Lab. Keep that environment's existing PyTorch and CUDA installation:

```sh
./isaaclab.sh -p -m pip install --no-deps /path/to/atlas-drive
```

Then configure an explicit actuator group:

```python
from atlas_drive_runtime.isaaclab import AtlasEnvelopeActuatorCfg

robot_cfg.actuators['atlas_joint'] = AtlasEnvelopeActuatorCfg(
    joint_names_expr=['your_rotary_joint_name'],
    profile_path='/absolute/path/to/your-envelope.json',
    update_dt_s=0.0025,  # actual interval between compute() calls
    temperature_C=25.,
    stiffness=50.,
    damping=2.,
)
```

The gains and update interval above are examples, not recommendations for a robot. Replace the existing actuator assignment for those joints; do not leave overlapping actuator groups. Keep the robot's physical armature, mass, joint limits and friction unless you have engineering evidence for a change.

The adapter computes requested effort from position/velocity targets and feed-forward effort, then applies the envelope and optional response lag. It sends explicit torque to PhysX and clears position/velocity targets. Existing actuator and simulator effort caps can impose further limits; inspect both rather than silently increasing them.

`temperature_C` stays at the configured value unless your environment updates the actuator's tensor. It is not a thermal simulation. Include `actuator.model_fault` in your environment's diagnostics or termination logic. `reset(env_ids)` clears the selected environments' response state and faults.

Run the integration smoke test with Isaac Lab's launcher:

```sh
./isaaclab.sh -p /path/to/atlas-drive/examples/isaaclab_smoke.py --headless
```

This checks construction, effort output and selective reset using the installed Isaac classes. It does not simulate a complete robot. This adapter has not been GPU/PhysX-qualified by this release; verify it in your environment before using it for a study. The CPU NumPy/PyTorch tests exercise the underlying model arithmetic independently.
