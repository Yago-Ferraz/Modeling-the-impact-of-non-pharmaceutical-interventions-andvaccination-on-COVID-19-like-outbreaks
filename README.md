# SEIRD Compartmental Model of COVID-19 Transmission

Coursework deliverable for **06280 — Mathematical-Computational Modeling
Applied to Epidemiology (T01)**, UFRPE, 2026.1.

This repository implements a **SEIRD** (Susceptible-Exposed-Infectious-Recovered-Deceased)
compartmental model, simulates a COVID-19-like outbreak in a closed population
of 1,000,000 individuals, and compares three scenarios: no intervention,
a temporary non-pharmaceutical intervention (NPI), and a vaccination campaign.

**Author:** Yago Moura Ferraz — UFRPE.

The full scientific report is in [`article/artigo.pdf`](article/artigo.pdf).

## Repository structure

```
epidemiologia/
├── src/
│   ├── seir_model.py        # SEIRD ODE model (SEIRDModel, SEIRDParams, Intervention, Vaccination)
│   ├── run_simulations.py   # Runs the 3 scenarios, saves CSVs + comparison figures
│   └── r0_sensitivity.py    # Sweeps R0, saves figures/r0_sensitivity.png
├── results/                 # Generated time-series CSVs (S,E,I,R,D,V per day)
├── figures/                 # Generated PNG figures
├── article/
│   └── artigo.pdf           # Scientific report (Scientific Reports style, 5 pp.)
├── requirements.txt
└── README.md
```

## How to run

Create and activate a virtual environment, then install the pinned
dependencies:

```bash
# Create the venv (once)
python -m venv venv

# Activate it
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (Git Bash):
source venv/Scripts/activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Then run the simulations:

```bash
cd src
python run_simulations.py     # baseline / NPI / vaccination scenarios
python r0_sensitivity.py       # R0 sensitivity sweep
```

Outputs are written to `../results/*.csv` and `../figures/*.png`.
The `venv/` directory is git-ignored (see `.gitignore`).

## Model summary

- Compartments: S, E, I, R, D, V (V = vaccinated, tracked separately from R
  so that infection-acquired immunity is not conflated with vaccine coverage).
- Parameters (defaults, early SARS-CoV-2 strain):
  - R0 = 2.5
  - Incubation period = 5.2 days (Lauer et al., 2020)
  - Infectious period = 7.0 days (Nishiura et al., 2020)
  - Case fatality rate = 1%
- Integrated numerically with `scipy.integrate.solve_ivp` (RK45).

## AI usage disclosure

Code, model design, and the accompanying article were developed with the
assistance of **Claude Code** (Anthropic, model `claude-sonnet-5`), an
AI coding agent. The author's own prompts guided the choice of disease
(COVID-19), scenario design (baseline/NPI/vaccination), and review/validation
of all generated equations and numerical results against the cited
literature. See the AI Usage Disclosure section in `article/artigo.pdf`
for details.

## License

Educational use — UFRPE course deliverable.
