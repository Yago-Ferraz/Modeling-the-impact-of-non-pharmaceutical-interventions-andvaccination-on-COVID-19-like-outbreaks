"""
SEIRD compartmental model for COVID-19 transmission dynamics.

Implements the classic Susceptible-Exposed-Infectious-Recovered-Deceased
(SEIRD) system of ordinary differential equations, following the standard
formulation in Keeling & Rohani, "Modeling Infectious Diseases in Humans
and Animals" (2008), extended with a case-fatality branch and support for
time-varying transmission (non-pharmaceutical interventions) and a
vaccination flow.

Equations
---------
dS/dt = -beta(t) * S * I / N - v(t) * S
dE/dt =  beta(t) * S * I / N - sigma * E
dI/dt =  sigma * E - gamma * I
dR/dt =  (1 - cfr) * gamma * I
dD/dt =  cfr * gamma * I
dV/dt =  v(t) * S

where N = S + E + I + R + D + V is the (constant) population size, and V
is a bookkeeping compartment for individuals immunized by vaccination
(never infected), kept separate from R (immunity acquired through
infection) so that infection-based outcomes such as attack rate are not
conflated with vaccine coverage.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SEIRDParams:
    """Epidemiological parameters for the SEIRD model.

    Default values are representative of early (pre-Alpha) SARS-CoV-2
    circulation and are drawn from the literature:

    - incubation_period (1/sigma) = 5.2 days [Lauer et al., Ann Intern Med 2020]
    - infectious_period (1/gamma) = 7.0 days [Nishiura et al., Int J Infect Dis 2020]
    - r0 = 2.5 (WHO situation report range: 2.0-3.5)
    - case_fatality_rate = 0.01 (early pandemic crude estimate, WHO 2020)
    """

    r0: float = 2.5
    incubation_period: float = 5.2
    infectious_period: float = 7.0
    case_fatality_rate: float = 0.01

    @property
    def sigma(self) -> float:
        return 1.0 / self.incubation_period

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period

    @property
    def beta(self) -> float:
        return self.r0 * self.gamma


@dataclass
class Intervention:
    """A non-pharmaceutical intervention that scales transmission (beta).

    start_day/end_day define the active window; `efficacy` is the fractional
    reduction applied to beta while active (0 = no effect, 1 = full stop of
    transmission).
    """

    start_day: float
    end_day: float
    efficacy: float

    def factor(self, t: float) -> float:
        if self.start_day <= t <= self.end_day:
            return 1.0 - self.efficacy
        return 1.0


@dataclass
class Vaccination:
    """Constant-rate vaccination campaign moving S directly to R.

    `daily_rate` is the fraction of the *remaining* susceptible pool
    vaccinated per day once the campaign starts.
    """

    start_day: float
    daily_rate: float

    def rate(self, t: float) -> float:
        return self.daily_rate if t >= self.start_day else 0.0


@dataclass
class SEIRDModel:
    population: float
    params: SEIRDParams = field(default_factory=SEIRDParams)
    interventions: list = field(default_factory=list)
    vaccination: Optional[Vaccination] = None

    def beta_t(self, t: float) -> float:
        beta = self.params.beta
        for iv in self.interventions:
            beta *= iv.factor(t)
        return beta

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        S, E, I, R, D, V = y
        N = self.population
        sigma = self.params.sigma
        gamma = self.params.gamma
        cfr = self.params.case_fatality_rate
        v = self.vaccination.rate(t) if self.vaccination else 0.0
        beta = self.beta_t(t)

        force_of_infection = beta * S * I / N
        vacc_flow = v * S

        dS = -force_of_infection - vacc_flow
        dE = force_of_infection - sigma * E
        dI = sigma * E - gamma * I
        dR = (1 - cfr) * gamma * I
        dD = cfr * gamma * I
        dV = vacc_flow
        return np.array([dS, dE, dI, dR, dD, dV])

    def run(self, t_span_days: float, e0: float = 1.0, i0: float = 0.0, dt: float = 0.1):
        """Integrate the model from t=0 to t=t_span_days.

        Returns a dict of time series arrays: t, S, E, I, R, D, V.
        """
        s0 = self.population - e0 - i0
        y0 = np.array([s0, e0, i0, 0.0, 0.0, 0.0])
        t_eval = np.arange(0.0, t_span_days + dt, dt)

        sol = solve_ivp(
            self._rhs,
            t_span=(0.0, t_span_days),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            max_step=1.0,
            rtol=1e-8,
            atol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        S, E, I, R, D, V = sol.y
        return {"t": sol.t, "S": S, "E": E, "I": I, "R": R, "D": D, "V": V}


def effective_r0(params: SEIRDParams, intervention_factor: float = 1.0) -> float:
    """Effective reproduction number under a given transmission reduction factor."""
    return params.r0 * intervention_factor
