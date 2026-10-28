"""On-device envelope interpolation and optional first-order torque response."""
import math
import torch

class TorchEnvelope:
    def __init__(self, profile, device="cpu", dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64")
        self.speed = torch.tensor(profile.speed.copy(), device=device, dtype=dtype)
        self.temperature = torch.tensor(profile.temperature.copy(), device=device, dtype=dtype)
        self.lower = torch.tensor(profile.lower.copy(), device=device, dtype=dtype)
        self.upper = torch.tensor(profile.upper.copy(), device=device, dtype=dtype)
        self.response_time_s = profile.response_time_s
        self.device, self.dtype = self.speed.device, dtype

    def evaluate(self, requested_torque_Nm, speed_rad_s, temperature_C=25.):
        request, speed, temperature = torch.broadcast_tensors(*[
            torch.as_tensor(x, device=self.device, dtype=self.dtype)
            for x in (requested_torque_Nm, speed_rad_s, temperature_C)])
        fault = (~torch.isfinite(request) | ~torch.isfinite(speed) | ~torch.isfinite(temperature)
                 | (speed < self.speed[0]) | (speed > self.speed[-1])
                 | (temperature < self.temperature[0]) | (temperature > self.temperature[-1]))
        speed = torch.nan_to_num(speed).clamp(self.speed[0],self.speed[-1])
        temperature = torch.nan_to_num(temperature).clamp(self.temperature[0],self.temperature[-1])
        i = (torch.searchsorted(self.speed, speed.contiguous(), right=True)-1).clamp(0,len(self.speed)-2)
        j = (torch.searchsorted(self.temperature, temperature.contiguous(), right=True)-1).clamp(0,len(self.temperature)-2)
        sx = (speed-self.speed[i])/(self.speed[i+1]-self.speed[i])
        tx = (temperature-self.temperature[j])/(self.temperature[j+1]-self.temperature[j])
        def interpolate(table):
            return ((1-tx)*((1-sx)*table[j,i]+sx*table[j,i+1])
                    +tx*((1-sx)*table[j+1,i]+sx*table[j+1,i+1]))
        lower, upper = interpolate(self.lower), interpolate(self.upper)
        torque = torch.where(fault, 0., torch.nan_to_num(request).clamp(lower,upper))
        return dict(torque_Nm=torque, lower_Nm=torch.where(fault,0.,lower), upper_Nm=torch.where(fault,0.,upper),
                    fault=fault, saturated=(torque != request) | fault)

class TorqueState:
    """One instance per actuator group. Step once per actual actuator update."""
    def __init__(self, envelope, shape):
        self.envelope = envelope
        self.torque = torch.zeros(shape,device=envelope.device,dtype=envelope.dtype)
        self.fault = torch.zeros(shape,device=envelope.device,dtype=torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.torque[ids] = 0.; self.fault[ids] = False

    def step(self, request, speed, temperature, dt):
        if isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:
            raise ValueError("dt must be the positive elapsed time between actuator updates")
        result = self.envelope.evaluate(request,speed,temperature)
        if result['torque_Nm'].shape != self.torque.shape:
            raise ValueError("Input shape must match the configured actuator group")
        self.fault |= result['fault']
        tau = self.envelope.response_time_s
        alpha = 1. if tau == 0 else -math.expm1(-dt/tau)
        next_torque = self.torque+alpha*(result['torque_Nm']-self.torque)
        # A shrinking speed/temperature limit overrides the response lag.
        self.torque.copy_(torch.where(self.fault,0.,next_torque.clamp(result['lower_Nm'],result['upper_Nm'])))
        result.update(torque_Nm=self.torque.clone(),fault=self.fault.clone(),
                      saturated=(self.torque != torch.as_tensor(request,device=self.envelope.device)) | self.fault)
        return result
