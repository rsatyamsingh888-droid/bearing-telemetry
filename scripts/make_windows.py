import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
PATTERN = re.compile(r"([HIOBC])_(\d+)_([012])\.csv$")
WINDOW = 4096
CHANNELS = ["Accelerometer", "Acoustic"]
CLASSES = ["healthy", "inner", "outer", "ball", "cage"]
FAULT_CLASS = {"I": 1, "O": 2, "B": 3, "C": 4}

# Rules fixed before training: per fault type, the highest-numbered bearing
# is the test bearing and the second highest the validation bearing.
VAL_BEARINGS = [4, 9, 14, 19]
TEST_BEARINGS = [5, 10, 15, 20]


def split_of(bearing):
    if bearing in TEST_BEARINGS:
        return "test"
    if bearing in VAL_BEARINGS:
        return "val"
    return "train"


def main():
    parts = {name: {"x": [], "y": [], "bearing": [], "state": []}
             for name in ["train", "val", "test"]}

    for path in sorted(RAW.glob("*.csv")):
        kind, bearing, state = PATTERN.search(path.name).groups()
        bearing, state = int(bearing), int(state)

        signal = pd.read_csv(path, usecols=CHANNELS)[CHANNELS].to_numpy(dtype=np.float32).T
        n = signal.shape[1] // WINDOW
        windows = signal[:, : n* WINDOW].reshape(len(CHANNELS), n, WINDOW).transpose(1, 0, 2)
        windows = windows - windows.mean(axis=2, keepdims=True)

        label = 0 if kind == "H" else FAULT_CLASS[kind]
        part = parts[split_of(bearing)]
        part["x"].append(windows)
        part["y"] += [label] * n
        part["bearing"] += [bearing] * n
        part["state"] += [state] * n
        print(f"{path.name}: {n} windows -> {split_of(bearing)} "
              f"mean {signal[0].mean():8.2f}  acoustic mean {signal[1].mean():7.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    train_x = np.concatenate(parts["train"]["x"])
    norm = {"channels": CHANNELS,
                "mean" : train_x.mean(axis=(0, 2)).tolist(),
                "std": train_x.std(axis=(0, 2)).tolist()}
    (OUT / "norm.json").write_text(json.dumps(norm, indent=2) + "\n")

    print()
    for name, part in parts.items():
        if part["x"]:
            x = np.concatenate(part["x"])
            y = np.array(part["y"])
            np.savez(OUT / f"{name}.npz", x=x, y=y,
                     bearing=np.array(part["bearing"]), state=np.array(part["state"]))
            counts = dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist()))
            print(f"{name:5s} {x.shape} {counts} bearings {sorted(set(part['bearing']))}")
    print("norm", norm)



if __name__ == "__main__":
    main()