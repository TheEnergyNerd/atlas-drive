"""Vectorized, signed torque limits; no extrapolation outside profile coverage."""
import numpy as np

def evaluate(profile, requested_torque_Nm, speed_rad_s, temperature_C=25.):
    request, speed, temperature = np.broadcast_arrays(
        np.asarray(requested_torque_Nm, dtype=float), np.asarray(speed_rad_s, dtype=float),
        np.asarray(temperature_C, dtype=float))
    fault = (~np.isfinite(request) | ~np.isfinite(speed) | ~np.isfinite(temperature)
             | (speed < profile.speed[0]) | (speed > profile.speed[-1])
             | (temperature < profile.temperature[0]) | (temperature > profile.temperature[-1]))
    speed = np.clip(np.nan_to_num(speed), profile.speed[0], profile.speed[-1])
    temperature = np.clip(np.nan_to_num(temperature), profile.temperature[0], profile.temperature[-1])
    i = np.clip(np.searchsorted(profile.speed, speed, side="right")-1, 0, len(profile.speed)-2)
    j = np.clip(np.searchsorted(profile.temperature, temperature, side="right")-1, 0, len(profile.temperature)-2)
    sx = (speed-profile.speed[i])/(profile.speed[i+1]-profile.speed[i])
    tx = (temperature-profile.temperature[j])/(profile.temperature[j+1]-profile.temperature[j])
    def interpolate(table):
        return ((1-tx)*((1-sx)*table[j,i]+sx*table[j,i+1])
                +tx*((1-sx)*table[j+1,i]+sx*table[j+1,i+1]))
    lower, upper = interpolate(profile.lower), interpolate(profile.upper)
    torque = np.where(fault, 0., np.clip(np.nan_to_num(request), lower, upper))
    return dict(torque_Nm=torque, lower_Nm=np.where(fault,0.,lower), upper_Nm=np.where(fault,0.,upper),
                fault=fault, saturated=(torque != request) | fault)
