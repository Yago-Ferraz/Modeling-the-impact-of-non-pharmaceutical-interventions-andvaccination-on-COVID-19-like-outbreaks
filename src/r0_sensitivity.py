"""
Sensitivity analysis: effect of R0 on peak infectious prevalence and final
attack rate, holding incubation/infectious periods fixed at COVID-19
literature values. Produces figures/r0_sensitivity.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seir_model import SEIRDModel, SEIRDParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "figures")

POPULATION = 1_000_000
T_SPAN = 500
E0 = 20
I0 = 5

R0_VALUES = np.arange(1.1, 5.01, 0.2)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    peaks = []
    attack_rates = []

    for r0 in R0_VALUES:
        params = SEIRDParams(r0=r0)
        model = SEIRDModel(population=POPULATION, params=params)
        series = model.run(T_SPAN, e0=E0, i0=I0)
        peaks.append(series["I"].max() / POPULATION * 100)
        attack_rates.append((series["R"][-1] + series["D"][-1]) / POPULATION * 100)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(R0_VALUES, peaks, "o-", color="#C44E52", label="Peak infectious prevalence")
    ax1.set_xlabel("Basic reproduction number (R0)")
    ax1.set_ylabel("Peak infectious prevalence (%)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.axvline(1.0, color="grey", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    ax2.plot(R0_VALUES, attack_rates, "s-", color="#4C72B0", label="Final attack rate")
    ax2.set_ylabel("Final attack rate (%)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    ax1.set_title("Sensitivity of outbreak severity to R0\n(SEIRD, incubation=5.2d, infectious=7.0d)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "r0_sensitivity.png"), dpi=200)
    plt.close(fig)

    print("R0\tPeak I (%)\tAttack rate (%)")
    for r0, p, a in zip(R0_VALUES, peaks, attack_rates):
        print(f"{r0:.1f}\t{p:.2f}\t\t{a:.2f}")


if __name__ == "__main__":
    main()
