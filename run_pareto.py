"""
run_pareto.py
-------------
Multi-objective optimization (NSGA-II, pymoo) of indoor hydroponic lettuce
room operation:

    decision variables : PPFD, photoperiod, air-temperature setpoint, RH setpoint
    objectives (min)   : energy (kWh/m2/cycle), water (L/m2/cycle), -fresh yield (kg/m2/cycle)

Outputs: results/pareto_front.csv, figures/pareto_front.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from cea_model import simulate_cycle, RoomParams

OUT_RES, OUT_FIG = "results", "figures"
os.makedirs(OUT_RES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

PARAMS = RoomParams()
MIN_YIELD = 2.0  # kg/m2/cycle: filter out non-viable corners of the front


class CEAProblem(ElementwiseProblem):
    def __init__(self):
        # x = [PPFD (umol/m2/s), photoperiod (h), T_air (C), RH (%)]
        super().__init__(
            n_var=4, n_obj=3, n_ieq_constr=0,
            xl=np.array([150.0, 12.0, 18.0, 55.0]),
            xu=np.array([450.0, 20.0, 26.0, 80.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        r = simulate_cycle(x[0], x[1], x[2], x[3], PARAMS)
        out["F"] = [r["energy_kwh"], r["water_l"], -r["yield_kg"]]


def main(seed=1):
    algo = NSGA2(pop_size=80)
    res = minimize(CEAProblem(), algo, get_termination("n_gen", 100), seed=seed, verbose=False)

    X, F = res.X, res.F
    df = pd.DataFrame({
        "ppfd": X[:, 0], "photoperiod_h": X[:, 1], "t_air_c": X[:, 2], "rh_pct": X[:, 3],
        "energy_kwh_m2": F[:, 0], "water_l_m2": F[:, 1], "yield_kg_m2": -F[:, 2],
    })
    df["kwh_per_kg"] = df.energy_kwh_m2 / df.yield_kg_m2
    df["l_per_kg"] = df.water_l_m2 / df.yield_kg_m2
    df = df.sort_values("energy_kwh_m2").reset_index(drop=True)
    df.to_csv(f"{OUT_RES}/pareto_front_all.csv", index=False)
    # The raw front includes degenerate corners (near-zero yield to minimise water/energy).
    # Keep only commercially meaningful solutions for reporting and plotting.
    n_all = len(df)
    df = df[df.yield_kg_m2 >= MIN_YIELD].reset_index(drop=True)
    df.to_csv(f"{OUT_RES}/pareto_front.csv", index=False)
    print(f"Raw Pareto set: {n_all}; kept {len(df)} with yield >= {MIN_YIELD} kg/m2/cycle")

    # reference (a typical "default" operating point) for context
    ref = simulate_cycle(250, 16, 22, 70, PARAMS)
    ref_kwh_kg = ref["energy_kwh"] / ref["yield_kg"]

    # figure: energy vs yield colored by water use; and efficiency vs setpoints
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = ax[0].scatter(df.energy_kwh_m2, df.yield_kg_m2, c=df.water_l_m2, cmap="viridis", s=28)
    ax[0].scatter([ref["energy_kwh"]], [ref["yield_kg"]], marker="*", s=180, c="red",
                  label="reference setpoints\n(250 µmol, 16 h, 22 °C, 70 %)")
    ax[0].set_xlabel("Energy (kWh m$^{-2}$ cycle$^{-1}$)")
    ax[0].set_ylabel("Fresh yield (kg m$^{-2}$ cycle$^{-1}$)")
    ax[0].set_title("Pareto front: energy vs yield")
    ax[0].legend(fontsize=8, loc="lower right")
    cb = plt.colorbar(sc, ax=ax[0]); cb.set_label("Water use (L m$^{-2}$ cycle$^{-1}$)")

    ax[1].scatter(df.yield_kg_m2, df.kwh_per_kg, c=df.ppfd, cmap="plasma", s=28)
    ax[1].axhline(ref_kwh_kg, color="red", ls="--", lw=1, label="reference setpoints")
    ax[1].set_xlabel("Fresh yield (kg m$^{-2}$ cycle$^{-1}$)")
    ax[1].set_ylabel("Energy per kg fresh mass (kWh kg$^{-1}$)")
    ax[1].set_title("Energy intensity (colour = PPFD)")
    ax[1].legend(fontsize=8)

    ax[2].scatter(df.rh_pct, df.water_l_m2, c=df.t_air_c, cmap="coolwarm", s=28)
    ax[2].set_xlabel("RH setpoint (%)")
    ax[2].set_ylabel("Water use (L m$^{-2}$ cycle$^{-1}$)")
    ax[2].set_title("Water use vs RH (colour = T$_{air}$)")
    plt.tight_layout()
    plt.savefig(f"{OUT_FIG}/pareto_front.png", dpi=160)
    plt.close()

    print(f"Pareto-optimal solutions: {len(df)}")
    print(df.describe().loc[["min", "max"]].round(2).to_string())
    knee = df.iloc[(df.kwh_per_kg - df.kwh_per_kg.min()).abs().argmin()]
    print("\nLowest energy-per-kg solution:")
    print(knee.round(2).to_string())
    print(f"\nReference (250 umol, 16 h, 22 C, 70 %): {ref['energy_kwh']:.1f} kWh, "
          f"{ref['water_l']:.1f} L, {ref['yield_kg']:.2f} kg, {ref_kwh_kg:.1f} kWh/kg")


if __name__ == "__main__":
    main()
