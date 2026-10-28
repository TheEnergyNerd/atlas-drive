"""Optional Isaac Lab 2.1.1 explicit-actuator adapter; import after AppLauncher."""
from dataclasses import MISSING
import math
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from .profile import EnvelopeProfile
from .torch_backend import TorchEnvelope, TorqueState

class AtlasEnvelopeActuator(ActuatorBase):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not math.isfinite(cfg.update_dt_s) or cfg.update_dt_s <= 0:
            raise ValueError('Set update_dt_s to the elapsed time between compute calls')
        envelope = TorchEnvelope(EnvelopeProfile.load(cfg.profile_path), device=self._device)
        self._drive = TorqueState(envelope,self.computed_effort.shape)
        self.temperature_C = self.computed_effort.new_full(self.computed_effort.shape,cfg.temperature_C)
        self.model_fault = self._drive.fault

    def reset(self, env_ids=None):
        self._drive.reset(env_ids)
        self.temperature_C[slice(None) if env_ids is None else env_ids] = self.cfg.temperature_C

    def compute(self, control_action, joint_pos, joint_vel):
        requested = joint_pos.new_zeros(joint_pos.shape)
        if control_action.joint_positions is not None:
            requested += self.stiffness*(control_action.joint_positions-joint_pos)
        if control_action.joint_velocities is not None or control_action.joint_positions is not None:
            target_velocity = 0. if control_action.joint_velocities is None else control_action.joint_velocities
            requested += self.damping*(target_velocity-joint_vel)
        if control_action.joint_efforts is not None:
            requested += control_action.joint_efforts
        self.computed_effort[:] = requested
        result = self._drive.step(requested,joint_vel,self.temperature_C,self.cfg.update_dt_s)
        # Retain any explicit actuator cap in addition to the supplied envelope.
        self.applied_effort[:] = result['torque_Nm'].clamp(-self.effort_limit,self.effort_limit)
        self._drive.torque.copy_(self.applied_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

@configclass
class AtlasEnvelopeActuatorCfg(ActuatorBaseCfg):
    class_type: type = AtlasEnvelopeActuator
    profile_path: str = MISSING
    update_dt_s: float = MISSING
    temperature_C: float = 25.
