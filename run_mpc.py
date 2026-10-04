"""
run_mpc.py
----------
Lag-aware model predictive control (MPC, CasADi) of nutrient-solution
temperature in a recirculating hydroponic reservoir, benchmarked against
rule-based on/off thermostats and against an MPC that ignores the delay.

Why this problem: root-zone temperature drives dissolved oxygen, nutrient
uptake and Pythium risk, and a chiller acts on the reservoir with a delay
(hydraulic transport, heat-exchanger and compressor response). Delays like
this are exactly what a predictive controller can account for and a
thermostat cannot.

Plant ("truth"):
    C dT/dt = UA_true (T_air - T) + Q_pump - Q_chiller(t - delay)
Controller model (MPC): same structure, but UA is 20 % lower than the truth
(model mismatch), and the air-temperature forecast is the nominal schedule
(no noise). Measurements carry sensor noise.

Outputs: results/mpc_metrics.csv, results/delay_sweep.csv,
         figures/mpc_timeseries.png, figures/mpc_delay_sweep.png
"""
import os
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_RES, OUT_FIG = "results", "figures"
os.makedirs(OUT_RES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ------------------------------------------------------------------ settings
DT_H = 1.0 / 12.0            # 5-minute control step (hours)
C_KWH_K = 100 * 4.186 / 3600  # 100 L reservoir thermal capacity, kWh/K
UA_TRUE = 0.036              # kW/K, reservoir <-> room air (truth)
UA_MODEL = 0.030             # kW/K, MPC model (20 % mismatch)
Q_PUMP = 0.10                # kW pump / root-zone heat gain
U_MAX = 0.80                 # kW thermal chiller capacity
COP = 3.0                    # constant chiller COP for electricity accounting
T_SET = 20.0                 # reservoir set-point, C
T_LO, T_HI = 18.0, 22.0      # acceptable root-zone band, C
N_HOR = 24                   # MPC horizon: 24 x 5 min = 2 h
SENSOR_SD = 0.05             # K
AIR_NOISE_SD = 0.25          # K (AR(1) noise on room air temperature)
SIM_DAYS = 3


def air_schedule(n_steps):
    """Room air temperature: 21 C lights-off, ~25 C lights-on (06:00-22:00), smooth transitions."""
    t_h = (np.arange(n_steps) * DT_H) % 24.0
    on = 1 / (1 + np.exp(-(t_h - 6.0) / 0.4)) - 1 / (1 + np.exp(-(t_h - 22.0) / 0.4))
    return 21.0 + 4.0 * on


def air_truth(n_steps, seed):
    rng = np.random.default_rng(seed)
    base = air_schedule(n_steps)
    noise = np.zeros(n_steps)
    for k in range(1, n_steps):
        noise[k] = 0.97 * noise[k - 1] + AIR_NOISE_SD * np.sqrt(1 - 0.97**2) * rng.standard_normal()
    return base + noise


# ------------------------------------------------------------------ plant
def plant_step(T, u_applied, T_air):
    return T + DT_H / C_KWH_K * (UA_TRUE * (T_air - T) + Q_PUMP - u_applied)


# ------------------------------------------------------------------ controllers
class OnOff:
    def __init__(self, band):
        self.band, self.on = band, False

    def act(self, T_meas, **_):
        if T_meas > T_SET + self.band:
            self.on = True
        elif T_meas < T_SET - self.band:
            self.on = False
        return U_MAX if self.on else 0.0


class MPC:
    """
    Receding-horizon MPC.  model_delay = number of 5-min steps the controller
    *believes* the actuator is delayed (0 = delay-unaware, = true delay -> lag-aware).
    Pending (already-issued, not-yet-effective) commands enter as known inputs.
    """

    def __init__(self, model_delay, w_track=60.0, w_energy=1.0, w_du=4.0, w_band=2000.0):
        self.d = model_delay
        N, d = N_HOR, model_delay
        opti = ca.Opti("conic")
        self.u = opti.variable(N)
        self.sh = opti.variable(N)  # slack above T_HI
        self.sl = opti.variable(N)  # slack below T_LO
        self.T0 = opti.parameter()
        self.past = opti.parameter(max(d, 1))     # commands issued but not yet effective
        self.uprev = opti.parameter()
        self.Taf = opti.parameter(N)

        a = DT_H / C_KWH_K
        T = self.T0
        cost = 0
        for i in range(N):
            u_eff = self.past[i] if i < d else self.u[i - d]
            T = T + a * (UA_MODEL * (self.Taf[i] - T) + Q_PUMP - u_eff)
            cost += w_track * (T - T_SET) ** 2
            cost += w_energy * self.u[i] / COP * 10.0
            du = self.u[i] - (self.uprev if i == 0 else self.u[i - 1])
            cost += w_du * du**2
            cost += w_band * (self.sh[i] ** 2 + self.sl[i] ** 2)
            opti.subject_to(self.sh[i] >= T - T_HI)
            opti.subject_to(self.sl[i] >= T_LO - T)
        opti.subject_to(self.sh >= 0)
        opti.subject_to(self.sl >= 0)
        opti.subject_to(opti.bounded(0, self.u, U_MAX))
        opti.minimize(cost)
        try:
            opti.solver("osqp", {"print_time": False}, {"verbose": False})
        except Exception:
            opti = None
        if opti is None:
            raise RuntimeError("OSQP plugin not available in this CasADi build")
        self.opti = opti
        self.hist = [0.0] * max(d, 1)   # last d issued commands (oldest first)
        self.last_u = 0.0

    def act(self, T_meas, Ta_forecast, **_):
        o = self.opti
        o.set_value(self.T0, T_meas)
        o.set_value(self.past, self.hist[-max(self.d, 1):] if self.d > 0 else [0.0])
        o.set_value(self.uprev, self.last_u)
        o.set_value(self.Taf, Ta_forecast)
        sol = o.solve()
        u0 = float(np.clip(sol.value(self.u)[0], 0.0, U_MAX))
        self.last_u = u0
        if self.d > 0:
            self.hist.append(u0)
            self.hist = self.hist[-self.d:]
        return u0


# ------------------------------------------------------------------ simulation
def simulate(controller, true_delay, n_days=SIM_DAYS, seed=7):
    n = int(n_days * 24 / DT_H)
    T_air = air_truth(n + N_HOR, seed)
    nominal = air_schedule(n + N_HOR)
    rng = np.random.default_rng(seed + 100)
    T = 21.5                                   # start slightly warm
    queue = [0.0] * true_delay                 # commands in transit
    Ts, Us, Ue = [], [], []
    for k in range(n):
        T_meas = T + SENSOR_SD * rng.standard_normal()
        u_cmd = controller.act(T_meas=T_meas, Ta_forecast=nominal[k:k + N_HOR])
        queue.append(u_cmd)
        u_eff = queue.pop(0) if true_delay > 0 else u_cmd
        T = plant_step(T, u_eff, T_air[k])
        Ts.append(T); Us.append(u_cmd); Ue.append(u_eff)
    return np.array(Ts), np.array(Us), np.array(Ue)


def metrics(Ts, Ue, n_days=SIM_DAYS):
    energy_kwh_day = Ue.sum() * DT_H / COP / n_days
    on = Ue > 0.05 * U_MAX
    starts_per_day = np.sum(on[1:] & ~on[:-1]) / n_days
    skip = int(2 / DT_H)                         # ignore first 2 h (initial transient)
    T = Ts[skip:]
    return dict(
        rmse_K=float(np.sqrt(np.mean((T - T_SET) ** 2))),
        pct_outside_band=float(100 * np.mean((T < T_LO) | (T > T_HI))),
        max_T=float(T.max()), min_T=float(T.min()),
        energy_kwh_el_day=float(energy_kwh_day),
        compressor_starts_day=float(starts_per_day),
    )


def make_controllers(true_delay):
    return {
        "On/off, ±0.5 K": OnOff(0.5),
        "On/off, ±1.0 K": OnOff(1.0),
        "MPC, delay-unaware": MPC(model_delay=0),
        "MPC, lag-aware": MPC(model_delay=true_delay),
    }


def main(true_delay=4):
    # --- main comparison at the base-case delay
    rows, traces = [], {}
    for name, ctrl in make_controllers(true_delay).items():
        Ts, Us, Ue = simulate(ctrl, true_delay)
        m = metrics(Ts, Ue); m["controller"] = name
        rows.append(m); traces[name] = (Ts, Ue)
    df = pd.DataFrame(rows).set_index("controller").round(3)
    df.to_csv(f"{OUT_RES}/mpc_metrics.csv")
    print(f"Base case: delay = {true_delay} steps = {true_delay*5} min, {SIM_DAYS} simulated days")
    print(df.to_string())

    # --- time series figure (day 2)
    n_day = int(24 / DT_H)
    sl = slice(n_day, 2 * n_day)
    t = np.arange(n_day) * DT_H
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"On/off, ±0.5 K": "tab:red", "On/off, ±1.0 K": "tab:orange",
              "MPC, delay-unaware": "tab:gray", "MPC, lag-aware": "tab:blue"}
    for name, (Ts, Ue) in traces.items():
        ax[0].plot(t, Ts[sl], label=name, color=colors[name], lw=1.4)
        ax[1].plot(t, Ue[sl], label=name, color=colors[name], lw=1.0)
    ax[0].axhspan(T_LO, T_HI, color="green", alpha=0.08, label="acceptable band")
    ax[0].axhline(T_SET, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel("Reservoir temperature (°C)")
    ax[0].set_title(f"Nutrient-solution temperature control, actuation delay = {true_delay*5} min (day 2)")
    ax[0].legend(ncol=3, fontsize=8, loc="upper left")
    ax[1].set_ylabel("Chiller output reaching\nreservoir (kW thermal)")
    ax[1].set_xlabel("Hour of day")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/mpc_timeseries.png", dpi=160); plt.close()

    # --- delay sweep: how does each controller degrade as the delay grows?
    sweep = []
    for d in [0, 2, 4, 6, 8]:
        for name, ctrl in make_controllers(d).items():
            Ts, Us, Ue = simulate(ctrl, d)
            m = metrics(Ts, Ue); m.update(controller=name, delay_min=d * 5)
            sweep.append(m)
    sw = pd.DataFrame(sweep)
    sw.round(3).to_csv(f"{OUT_RES}/delay_sweep.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in colors:
        s = sw[sw.controller == name]
        ax[0].plot(s.delay_min, s.rmse_K, "o-", label=name, color=colors[name])
        ax[1].plot(s.delay_min, s.pct_outside_band, "o-", label=name, color=colors[name])
        ax[2].plot(s.delay_min, s.energy_kwh_el_day, "o-", label=name, color=colors[name])
    ax[0].set_ylabel("RMSE vs set-point (K)")
    ax[1].set_ylabel("Time outside 18–22 °C (%)")
    ax[2].set_ylabel("Chiller electricity (kWh/day)")
    for a in ax:
        a.set_xlabel("Actuation delay (min)")
    ax[0].legend(fontsize=8)
    plt.suptitle("Effect of actuation delay on control performance")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/mpc_delay_sweep.png", dpi=160); plt.close()
    print("\nDelay sweep (RMSE K / % outside band / kWh per day):")
    print(sw.pivot(index="controller", columns="delay_min", values="rmse_K").round(2).to_string())
    print(sw.pivot(index="controller", columns="delay_min", values="pct_outside_band").round(1).to_string())
    print(sw.pivot(index="controller", columns="delay_min", values="energy_kwh_el_day").round(2).to_string())


if __name__ == "__main__":
    main()
