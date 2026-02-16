import argparse

from cxotime import CxoTime
from kadi.events import scs107s

from storms import SolarWind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("radmon_disable_time", type=str)
    parser.add_argument("return_to_science_time", type=str)

    args = parser.parse_args()

    start = CxoTime(args.radmon_disable_time)
    stop = CxoTime(args.return_to_science_time)

    s107s = scs107s.filter(start, stop)

    sw = SolarWind(
        start,
        stop,
        use_browse=True,
        get_states=True,
    )

    potential_fluence = sw.get_ace_p3_fluence(start, stop)
    actual_fluence = sw.get_ace_p3_fluence(start, stop, planned=False)

    print(f"RADMON Enable: {start}")
    print(f"Return to science: {stop}")
    print("")

    for s107 in s107s:
        print(f"SCS 107 occured at {s107.start}.")

    print("")
    print(f"Potential fluence: {potential_fluence:.2f} {chr(0x00D7)} 10{chr(0x2079)}")
    print(f"Actual fluence: {actual_fluence:.2f} {chr(0x00D7)} 10{chr(0x2079)}")
    print(
        f"Fluence avoided: {potential_fluence - actual_fluence:.2f} {chr(0x00D7)} 10{chr(0x2079)}"
    )


if __name__ == "__main__":
    main()
