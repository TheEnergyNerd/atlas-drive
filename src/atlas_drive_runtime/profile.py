"""A strict, portable envelope contract in output-shaft SI units."""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = {"schema", "name", "speed_rad_s", "temperature_C", "lower_torque_Nm",
          "upper_torque_Nm", "response_time_s", "description"}

class EnvelopeProfile:
    def __init__(self, document):
        if not isinstance(document, dict) or set(document) != FIELDS:
            raise ValueError("Supply exactly the documented envelope fields")
        if document["schema"] != "atlas-drive-envelope-v1":
            raise ValueError("Unsupported profile schema")
        if not isinstance(document["name"], str) or not 1 <= len(document["name"]) <= 120:
            raise ValueError("Invalid profile name")
        if not isinstance(document["description"], str):
            raise ValueError("Description must be text")
        self._document = json.loads(json.dumps(document, allow_nan=False))
        self.speed = self._array("speed_rad_s")
        self.temperature = self._array("temperature_C")
        for axis in (self.speed, self.temperature):
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
                raise ValueError("Axes must be finite, strictly increasing and contain at least two points")
        if not self.speed[0] < 0 < self.speed[-1]:
            raise ValueError("Speed coverage must include both signs and zero")
        self.lower = self._array("lower_torque_Nm")
        self.upper = self._array("upper_torque_Nm")
        shape = (len(self.temperature), len(self.speed))
        if self.lower.shape != shape or self.upper.shape != shape:
            raise ValueError("Torque tables must have shape [temperature, speed]")
        if np.any(self.lower > 0) or np.any(self.upper < 0):
            raise ValueError("Every envelope interval must contain zero torque")
        tau = document["response_time_s"]
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not np.isfinite(tau) or tau < 0:
            raise ValueError("Response time must be a finite nonnegative number")
        self.response_time_s = float(tau)
        self.sha256 = hashlib.sha256(json.dumps(self._document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _array(self, key):
        value = np.array(self._document[key], dtype=float, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("Profile values must be finite")
        value.flags.writeable = False
        return value

    @property
    def document(self):
        return json.loads(json.dumps(self._document))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise ValueError("Profile exceeds 1 MB")
        return cls(json.loads(path.read_text()))
