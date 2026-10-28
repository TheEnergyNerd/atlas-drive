# Envelope model format

Schema: `atlas-drive-envelope-v1`. The parser accepts exactly these fields:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable profile name |
| `description` | Source and scope; identify synthetic examples |
| `speed_rad_s` | Strictly increasing signed output-shaft speeds, including coverage on both sides of zero |
| `temperature_C` | Strictly increasing externally supplied temperatures |
| `lower_torque_Nm` | Minimum torque table, rows = temperatures, columns = speeds |
| `upper_torque_Nm` | Maximum torque table with the same shape |
| `response_time_s` | Nonnegative first-order torque-response time; zero disables lag |
| `schema` | The schema identifier above |

Each torque interval must contain zero. Every number must be finite. A profile may be loaded from a local JSON file of at most 1 MB. Profiles cannot contain executable code or arbitrary extra data. See the complete [synthetic example](../examples/synthetic-envelope.json).

The runtime uses bilinear interpolation, including at the endpoints. It never extrapolates: speed or temperature outside the profile, or a nonfinite demand/state, produces `fault=true` and zero commanded torque. `saturated` is true whenever requested torque is not delivered, including a fault. Check the fault flag even when requesting zero torque.

`TorqueState` latches faults until reset, adds the optional response lag, then clamps again to the current envelope. A decreasing speed/temperature limit therefore takes precedence over lag. Resetting one batch element does not reset the others. Pass the actual elapsed time between updates; calling twice per physics step changes the response.

The profile fingerprint hashes its canonical JSON representation. Loading and saving an unchanged document retains that fingerprint. It identifies an artifact; it does not certify calibration, sample provenance or mechanical qualification.

This format describes externally characterized behavior. It contains no default motor physics, manufacturing design generator or temperature integrator. A profile that came from an inaccurate source model remains inaccurate. In particular, do not interpret an interpolated envelope as independently measured hardware performance.
