"""
cea_model.py
------------
Simplified, per-m2-of-floor model of an indoor hydroponic lettuce room
(sole-source LED lighting, recirculating NFT/DWC-type nutrient solution).

It couples, in one place:
  * canopy transpiration  -> latent load the HVAC must remove
  * lighting + envelope   -> sensible load
  * HVAC electricity      -> energy
  * a light-use-efficiency growth model -> yield
  * water use             -> transpired water that must be replaced

IMPORTANT: this is an illustrative prototype. Parameters are plausible
literature-order-of-magnitude values, NOT calibrated to a specific facility.
The point is the workflow (physics-based model -> multi-objective
optimization -> predictive control), not the absolute numbers.
"""
from dataclasses import dataclass
import numpy as np

LAMBDA_WH_PER_G = 0.68  # latent heat of vaporization of water, Wh per g (~2450 kJ/kg)


def sat_vp_kpa(t_c):
    """Saturation vapour pressure (kPa), Tetens equation."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def vpd_kpa(t_c, rh_pct):
    return sat_vp_kpa(t_c) * (1.0 - rh_pct / 100.0)


@dataclass
class RoomParams:
    # --- environment / envelope ---
    t_out: float = 30.0          # outdoor air temperature, C (hot-climate case)
    ua_env: float = 0.8          # envelope conductance, W/K per m2 floor
    # --- lighting ---
    led_efficacy: float = 2.5    # umol/J (photons per electrical joule)
    # --- HVAC ---
    cop0: float = 3.2            # cooling COP at 22 C room setpoint
    cop_slope: float = 0.03      # fractional COP gain per K of higher room T
    fan_w: float = 4.0           # air-handling / circulation fans, W per m2 floor
    # --- crop (lettuce, illustrative) ---
    cycle_days: int = 28
    w0: float = 2.0              # initial dry mass, g/m2 (seedlings)
    sla: float = 0.030           # m2 leaf per g dry mass (effective, incl. partitioning)
    lai_max: float = 4.0
    k_ext: float = 0.7           # canopy light-extinction coefficient
    lue0: float = 1.35           # g dry mass per mol intercepted PAR (low-light limit)
    ppfd_half: float = 500.0     # umol/m2/s where LUE drops to ~50% (diminishing returns)
    t_opt: float = 22.0
    dm_fraction: float = 0.05    # dry matter / fresh matter
    # --- transpiration ---
    c_trans: float = 15.0        # g/m2/h per (kPa VPD * LAI), lights-on
    night_factor: float = 0.10   # night transpiration relative to day


def f_temp(t_c, p: RoomParams):
    """Bell-shaped temperature response (1 at optimum)."""
    return np.exp(-((t_c - p.t_opt) / 6.0) ** 2)


def f_vpd(v):
    """Penalty for very low or very high VPD (stomatal / tipburn / disease stress)."""
    v = np.asarray(v, dtype=float)
    low = np.clip((v - 0.3) / 0.3, 0, 1)       # <0.3 kPa -> strong penalty
    high = np.clip(1.0 - (v - 1.2) / 1.0, 0, 1)  # >1.2 kPa -> decline
    return np.minimum(low, high) * 0.9 + 0.1 if np.ndim(v) else float(min(low, high) * 0.9 + 0.1)


def f_photoperiod(h):
    """Mild penalty for very long photoperiods (>20 h): tipburn / injury risk."""
    return 1.0 if h <= 20.0 else max(0.0, 1.0 - 0.15 * (h - 20.0))


def cop(t_air, p: RoomParams):
    return p.cop0 * (1.0 + p.cop_slope * (t_air - 22.0))


def simulate_cycle(ppfd, photoperiod_h, t_air, rh_pct, p: RoomParams = RoomParams()):
    """
    Simulate one production cycle at daily resolution.

    Decision variables (operation setpoints):
      ppfd            : canopy PPFD while lights are on, umol/m2/s
      photoperiod_h   : lights-on hours per day
      t_air           : room air temperature setpoint, C
      rh_pct          : room relative-humidity setpoint, %

    Returns dict with energy (kWh/m2/cycle), water (L/m2/cycle),
    fresh yield (kg/m2/cycle), and diagnostic series.
    """
    dli = ppfd * photoperiod_h * 3600.0 / 1e6              # mol/m2/day
    vpd = vpd_kpa(t_air, rh_pct)
    # LUE with diminishing returns at high PPFD
    lue = p.lue0 / (1.0 + ppfd / p.ppfd_half)
    stress = f_temp(t_air, p) * f_vpd(vpd) * f_photoperiod(photoperiod_h)

    w = p.w0
    water_g = 0.0
    e_light = e_hvac = 0.0
    lai_series, w_series = [], []

    p_light_w = ppfd / p.led_efficacy                       # electrical W/m2 while on
    cop_val = cop(t_air, p)

    for _ in range(p.cycle_days):
        lai = min(p.lai_max, p.sla * w)
        interception = 1.0 - np.exp(-p.k_ext * lai)
        dw = lue * dli * interception * stress              # g/m2/day
        # transpiration: day (lights on) + night
        e_day = p.c_trans * lai * vpd * photoperiod_h       # g/m2/day
        e_night = p.c_trans * lai * vpd * p.night_factor * (24.0 - photoperiod_h)
        water_g += e_day + e_night

        # loads (Wh/m2/day)
        q_light_wh = p_light_w * photoperiod_h              # all lighting power ends up as heat
        q_env_wh = p.ua_env * (p.t_out - t_air) * 24.0
        q_lat_wh = (e_day + e_night) * LAMBDA_WH_PER_G
        q_cool_wh = max(0.0, q_light_wh + q_env_wh + q_lat_wh)
        e_light += q_light_wh / 1000.0
        e_hvac += (q_cool_wh / cop_val + p.fan_w * 24.0) / 1000.0

        w += dw
        lai_series.append(lai)
        w_series.append(w)

    fresh_kg = w / p.dm_fraction / 1000.0
    return dict(
        energy_kwh=e_light + e_hvac,
        energy_light_kwh=e_light,
        energy_hvac_kwh=e_hvac,
        water_l=water_g / 1000.0,
        yield_kg=fresh_kg,
        dli=dli, vpd=vpd,
        lai=np.array(lai_series), dry_mass=np.array(w_series),
    )


if __name__ == "__main__":
    r = simulate_cycle(ppfd=250, photoperiod_h=16, t_air=22, rh_pct=70)
    print({k: (round(v, 3) if np.isscalar(v) else '...') for k, v in r.items()})
    print("kWh per kg fresh:", round(r['energy_kwh'] / r['yield_kg'], 2))
