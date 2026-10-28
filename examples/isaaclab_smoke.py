"""Run through isaaclab.sh after installing atlas-drive-runtime into its environment."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
try:
    import torch
    from isaaclab.utils.types import ArticulationActions
    from atlas_drive_runtime.isaaclab import AtlasEnvelopeActuator, AtlasEnvelopeActuatorCfg
    cfg=AtlasEnvelopeActuatorCfg(joint_names_expr=['test_joint'],
        profile_path=str(Path(__file__).parent/'synthetic-envelope.json'),
        update_dt_s=.0025, stiffness=0., damping=0.)
    drive=AtlasEnvelopeActuator(cfg,joint_names=['test_joint'],joint_ids=slice(None),num_envs=2,device=args.device)
    velocity=torch.full((2,1),10.,device=args.device)
    position=torch.zeros_like(velocity)
    for _ in range(100):
        result=drive.compute(ArticulationActions(joint_efforts=torch.full_like(velocity,25.)),position,velocity)
    assert torch.allclose(result.joint_efforts,torch.full_like(velocity,20.),atol=1e-4)
    drive.reset(torch.tensor([0],device=args.device))
    assert drive._drive.torque[0].item()==0 and drive._drive.torque[1].item()>19.9
    print('Isaac actuator construction, effort output and selective reset passed.')
finally:
    launcher.app.close()
