"""
Runs three COVID-19 SEIRD scenarios (baseline, NPI/lockdown, vaccination),
saves time series to results/*.csv, generates comparison figures to
figures/*.png, and prints summary statistics used in the accompanying
scientific article.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seir_model import SEIRDModel, SEIRDParams, Intervention, Vaccination

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

POPULATION = 1_000_000
T_SPAN = 365
E0 = 20
I0 = 5


def peak_stats(series: dict) -> dict:
    i_peak_idx = int(np.argmax(series["I"]))
    # Attack rate counts only infection-acquired immunity (R + D), excluding
    # the vaccinated compartment V, so vaccine coverage is not conflated
    # with the fraction of the population actually infected.
    infected_total = series["R"][-1] + series["D"][-1]
    return {
        "peak_infectious": series["I"][i_peak_idx],
        "peak_day": series["t"][i_peak_idx],
        "total_deaths": series["D"][-1],
        "attack_rate": infected_total / POPULATION,
        "vaccinated_fraction": series["V"][-1] / POPULATION,
    }


def save_csv(series: dict, name: str):
    df = pd.DataFrame(series)
    df.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)


def run_baseline():
    model = SEIRDModel(population=POPULATION, params=SEIRDParams())
    return model.run(T_SPAN, e0=E0, i0=I0)


def run_npi():
    params = SEIRDParams()
    interventions = [Intervention(start_day=30, end_day=120, efficacy=0.60)]
    model = SEIRDModel(population=POPULATION, params=params, interventions=interventions)
    return model.run(T_SPAN, e0=E0, i0=I0)


def run_vaccination():
    params = SEIRDParams()
    vacc = Vaccination(start_day=60, daily_rate=0.01)
    model = SEIRDModel(population=POPULATION, params=params, vaccination=vacc)
    return model.run(T_SPAN, e0=E0, i0=I0)


def plot_scenario_comparison(baseline, npi, vaccination):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    scenarios = [
        ("Baseline (no intervention)", baseline),
        ("NPI: 60% transmission reduction\n(days 30-120)", npi),
        ("Vaccination from day 60\n(1%/day of susceptibles)", vaccination),
    ]
    colors = {"S": "#4C72B0", "E": "#DD8452", "I": "#C44E52", "R": "#55A868", "D": "#333333", "V": "#8172B2"}

    for ax, (title, series) in zip(axes, scenarios):
        for comp in ["S", "E", "I", "R", "D", "V"]:
            ax.plot(series["t"], series[comp] / POPULATION * 100, label=comp, color=colors[comp])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (days)")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Population (%)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("SEIRD dynamics of a COVID-19-like outbreak under three scenarios")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "scenario_comparison.png"), dpi=200)
    plt.close(fig)


def plot_infectious_curve(baseline, npi, vaccination):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(baseline["t"], baseline["I"], label="Baseline", color="#C44E52")
    ax.plot(npi["t"], npi["I"], label="NPI (60% reduction, days 30-120)", color="#4C72B0")
    ax.plot(vaccination["t"], vaccination["I"], label="Vaccination (from day 60)", color="#55A868")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infectious individuals")
    ax.set_title("Effect of interventions on the epidemic curve (\"flattening the curve\")")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "infectious_curve_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    baseline = run_baseline()
    npi = run_npi()
    vaccination = run_vaccination()

    save_csv(baseline, "baseline")
    save_csv(npi, "npi_intervention")
    save_csv(vaccination, "vaccination")

    plot_scenario_comparison(baseline, npi, vaccination)
    plot_infectious_curve(baseline, npi, vaccination)

    print("Scenario summary (population = {:,})".format(POPULATION))
    print("-" * 60)
    for name, series in [("Baseline", baseline), ("NPI", npi), ("Vaccination", vaccination)]:
        stats = peak_stats(series)
        print(
            f"{name:12s} | peak I = {stats['peak_infectious']:>10,.0f} "
            f"(day {stats['peak_day']:>5.1f}) | "
            f"total deaths = {stats['total_deaths']:>8,.0f} | "
            f"attack rate = {stats['attack_rate']*100:5.1f}% | "
            f"vaccinated = {stats['vaccinated_fraction']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
