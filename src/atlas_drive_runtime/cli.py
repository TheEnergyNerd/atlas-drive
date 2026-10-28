import argparse
import json
from . import EnvelopeProfile, evaluate

def main():
    parser = argparse.ArgumentParser(description='Evaluate a supplied actuator envelope locally.')
    parser.add_argument('profile')
    parser.add_argument('--torque', type=float, required=True, help='Requested output torque, Nm')
    parser.add_argument('--speed', type=float, required=True, help='Output speed, rad/s')
    parser.add_argument('--temperature', type=float, default=25., help='Externally supplied winding temperature, Celsius')
    args = parser.parse_args()
    profile = EnvelopeProfile.load(args.profile)
    result = evaluate(profile,args.torque,args.speed,args.temperature)
    print(json.dumps(dict(profile_sha256=profile.sha256,**{key:value.item() for key,value in result.items()}),allow_nan=False,indent=2))
    return 2 if bool(result['fault']) else 0

if __name__ == '__main__':
    raise SystemExit(main())
