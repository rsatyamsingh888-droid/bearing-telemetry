import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAW = Path("data/raw")
PATTERN = re.compile(r"([HIOBC])_(\d+)_([012])\.csv$")
STATES = ["healthy", "developing", "faulty"]
FAULTS = {"I": "inner race", "O": "outer race", "B": "ball", "C": "cage"}


def main():
    rows = []
    fault_of = {}
    for path in sorted(RAW.glob("*.csv")):
        kind, bearing, state = PATTERN.search(path.name).groups()
        bearing = int(bearing)
        if kind != "H":
            fault_of[bearing] = FAULTS[kind]
        df = pd.read_csv(path, usecols=["Accelerometer", "Acoustic"])
        rows.append({"bearing": bearing, "state": STATES[int(state)],
                     "vib": df["Accelerometer"].std(), "acoustic": df["Acoustic"].std()})
        print(f"read {path.name}")

    table = pd.DataFrame(rows)
    vib = table.pivot(index="bearing", columns="state", values="vib")[STATES]
    vib["fault"] = vib.index.map(fault_of)

    lines = ["# Vibration level per recording","",
            "Standard deviation of the accelerometer signal, per bearing and state.", "",
            "| Bearing | Fault | Healthy | Developing | Faulty |", "|---|---|---|---|---|"]
    for bearing, r in vib.iterrows():
        lines.append(f"| {bearing} | {r['fault']} | {r['healthy']:.1f} | "
                     f"{r['developing']:.1f} | {r['faulty']:.1f} |")


    louder = (vib["developing"] > vib["faulty"]).sum()
    above = (vib[["developing", "faulty"]].min(axis=1) > vib["healthy"]).sum()
    lines += ["", f"Developing louder than faulty: {louder} of {len(vib)} bearings",
              f"Both damaged states louder than healthy: {above} of {len(vib)} bearings."]
    Path("docs/survey.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, fault in zip(axes, FAULTS.values()):
        for bearing, r in vib[vib["fault"] == fault].iterrows():
            ax.plot(STATES, r[STATES].astype(float), marker="o", label=f"bearing {bearing}")
        ax.set_title(fault)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("vibration std")
    fig.tight_layout()
    fig.savefig("docs/survey.png", dpi=120)

    print("\n".join(lines[-2:]))
    print("wrote docs/survey.md and docs/survey.png")



if __name__ == "__main__":
    main()