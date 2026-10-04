"""run_all.py - runs the whole prototype in one go (model check, NSGA-II, MPC)."""
import run_pareto
import run_mpc
import cea_model

print("=== 1/3 Model sanity check ===")
r = cea_model.simulate_cycle(ppfd=250, photoperiod_h=16, t_air=22, rh_pct=70)
print(f"Energy {r['energy_kwh']:.1f} kWh/m2 | Water {r['water_l']:.1f} L/m2 | Yield {r['yield_kg']:.2f} kg/m2")

print("\n=== 2/3 NSGA-II optimization ===")
run_pareto.main()

print("\n=== 3/3 MPC vs on/off control ===")
run_mpc.main()

print("\nDone. Check the 'figures' and 'results' folders.")
