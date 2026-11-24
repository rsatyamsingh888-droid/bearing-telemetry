import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

RAW = Path("data/raw")
FS = 42_000

def load_vibration(name):
    signal = pd.read_csv(RAW / f"{name}.csv", usecols=["Accelerometer"])["Accelerometer"].to_numpy()
    return signal - signal.mean()


def spectrum(signal):
    freqs, power = welch(signal, fs=FS, nperseg=4096)
    return freqs, power


def main():
    files = {"healthy": "H_1_0", "developing": "I_1_1", "faulty": "I_1_2"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharey="col")

    for row, (state, name) in enumerate(files.items()):
        x = load_vibration(name)
        t = np.arange(len(x)) / FS

        show = t < 0.1
        axes[row, 0].plot(t[show] * 1000, x[show], linewidth=0.5)
        axes[row, 0].set_ylabel(f"{state}\nacceleration")

        freqs, amp = spectrum(x)
        axes[row, 1].semilogy(freqs, amp, linewidth=0.5)
        axes[row, 1].set_xlim(0, 5000)

        print(f"{state:10s} {name}  std {x.std():.3f}")

    axes[0, 0].set_title("vibration, first 100 ms")
    axes[0, 1].set_title("frequency spectrum, 0-5 kHz")
    axes[2, 0].set_xlabel("time (ms)")
    axes[2, 1].set_xlabel("frequency (Hz)")
    fig.suptitle("Bearing 1 (inner-race fault), accelerometer")
    fig.tight_layout()

    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/bearing1_signals.png", dpi=120)
    print("wrote docs/bearing1_signals.png")
        



if __name__ == "__main__":
    main()