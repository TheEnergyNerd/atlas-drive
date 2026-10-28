# Contributing

Open an issue for model-format or integration problems, ideally with a synthetic reproducer. Include the Python, PyTorch, Isaac Lab and GPU versions relevant to the issue.

Do not attach customer workloads, private actuator profiles, measured calibration, raw bench data, manufacturing drawings, credentials or model weights. Replace them with synthetic values that reproduce the problem. Issue and pull-request contents are public.

Run `python -m pytest -q` before proposing a change. Changes to interpolation, saturation, sign conventions, reset behavior or units need a focused regression test. State whether an Isaac change was tested in an actual simulator or only against the CPU runtime.

Contributions are under the repository's MIT license. Dependencies retain their own licenses.
