"""Generate the 2026 input data from the baseline (2020) files.

Run from this directory: python make_2026.py

Changes from the baseline, all agreed for the reviewer response:
  hydro          production cost 1.4 -> 1.6 (inflation, +14.3%)
  thermal-solid  production cost 4.4 -> 5.0 (inflation only, same +14.3%;
                 the source's 14.7 includes a CO2 charge, which we leave
                 out as a transfer within government, not a production
                 cost), CO2 0.82 -> 1.6 kg/kWh (poorer lignite),
                 lower bound 20% -> 40% of capacity (rounded to MW)
  thermal-gas    production cost 12.0 -> 13.7 (inflation only, +14.3%;
                 no CO2 charge, for the same reason)
  solar          production cost unchanged at 6.6; CO2 0.030 -> 0.048,
                 and the hourly CO2 profile, which is what run.py
                 actually uses for solar, scaled by the same 1.6x
  all plants     environment and sustainability cost +10% (inflation)
Demand and solar production are unchanged (run.py reads demand.csv
from ../data/ in every scenario).
"""
import pandas as pd

INFLATION = 1.6 / 1.4          # hydro's stated increase, applied to fuels
COST = {"hydro": 1.6,
        "thermal-solid": round(4.4 * INFLATION, 1),   # 5.0
        "thermal-gas": round(12.0 * INFLATION, 1),    # 13.7
        "solar": 6.6}
LIGNITE_CO2 = 1.6
LIGNITE_LB_FRACTION = 0.4
SOLAR_CO2_FACTOR = 0.048 / 0.030
ENV_SUS_FACTOR = 1.1

for solar in (0, 10, 50, 100):
    pi = pd.read_csv(f"../plant_info_{solar}.csv", encoding="utf-8-sig")
    pi["production_cost"] = pi["type"].map(COST)
    lig = pi["type"] == "thermal-solid"
    pi["CO2"] = pi["CO2"].astype(float)
    pi.loc[lig, "CO2"] = LIGNITE_CO2
    pi.loc[lig, "lower_bound"] = (LIGNITE_LB_FRACTION
                                  * pi.loc[lig, "upper_bound"]).round().astype(int)
    sol = pi["type"] == "solar"
    pi.loc[sol, "CO2"] = round(0.030 * SOLAR_CO2_FACTOR, 4)
    for col in ("environment_cost", "sustainability_cost"):
        pi[col] = (pi[col] * ENV_SUS_FACTOR).round(4)
    pi.to_csv(f"plant_info_{solar}.csv", index=False, encoding="utf-8-sig")

sp = pd.read_csv("../solar_production_and_CO2_per_hour.csv", encoding="utf-8-sig")
sp["CO2"] = (sp["CO2"] * SOLAR_CO2_FACTOR).round(4)
sp.to_csv("solar_production_and_CO2_per_hour.csv", index=False, encoding="utf-8-sig")
