"""
=============================================================================
 run_validation.py  —  Jade Valley Flood Simulation  (75%)
=============================================================================
 PURPOSE:
   Compares your simulation results against a real typhoon event to check
   how well the model matches reality.

 HOW TO USE (3 steps):
   1. Run flood_simulation_75%.py with the typhoon's rainfall settings.
   2. Open  Results/data/quantitative_results_<scenario>.txt
      and copy  "Peak flood extent %"  and  "Peak maximum depth mm".
   3. Paste those two numbers below where marked, then run this script.

 RUN:
   python Main/run_validation.py
=============================================================================
"""

import importlib.util
from pathlib import Path

# ── Load the simulation module (handles the % in the filename) ───────────────
_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "sim75", _root / "Main" / "flood_simulation_75%.py"
)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

validate_against_event = _sim.validate_against_event
DATA_OUT               = _sim.DATA_OUT

# =============================================================================
#  ↓↓  EDIT THESE TWO NUMBERS AFTER RUNNING THE SIMULATION  ↓↓
# =============================================================================

EVENT_NAME = "Typhoon Vinta 2017"   # must match a key in REFERENCE_EVENTS
                                     # Options:
                                     #   "Typhoon Vinta 2017"
                                     #   "Typhoon Pablo 2012"
                                     #   "Typhoon Odette 2021"
                                     #   "Habagat Heavy Rain Episode"

# Copy these from Results/data/quantitative_results_<scenario>.txt
SIMULATED_PEAK_FLOODED_PCT = 0.0    # ← replace with your value  e.g. 34.52
SIMULATED_PEAK_DEPTH_MM    = 0.0    # ← replace with your value  e.g. 812.3

# =============================================================================
#  ↑↑  STOP EDITING HERE  ↑↑
# =============================================================================

if __name__ == "__main__":
    if SIMULATED_PEAK_FLOODED_PCT == 0.0 and SIMULATED_PEAK_DEPTH_MM == 0.0:
        print("\n  [!] You haven't filled in the simulated values yet.")
        print("      1. Run flood_simulation_75%.py with Typhoon Vinta settings:")
        print("           Rainfall : 160 mm")
        print("           Duration : 12 h")
        print("           Pattern  : Burst")
        print("           Prevention measures: OFF")
        print("      2. Open Results/data/quantitative_results_<scenario>.txt")
        print("      3. Copy 'Peak flood extent %' and 'Peak maximum depth mm'")
        print("      4. Paste them into SIMULATED_PEAK_FLOODED_PCT and")
        print("         SIMULATED_PEAK_DEPTH_MM above, then re-run this script.")
    else:
        validate_against_event(
            event_name=EVENT_NAME,
            simulated_peak_flooded_pct=SIMULATED_PEAK_FLOODED_PCT,
            simulated_peak_depth_mm=SIMULATED_PEAK_DEPTH_MM,
            out_dir=DATA_OUT,
        )
        print(f"\n  Done. Check Results/data/ for the validation report.")
