"""Which demand days does the model admit a schedule for at all?

data/demand.csv holds four hourly demand curves: two winter days
(19-Jan-21, the paper's baseline, and 09-Jan-26) and two summer days
(14-Jul-21 and 03-Aug-26).  We wanted the summer days as a sensitivity
study, but the LP in run.py has no feasible solution for either of them.
This script shows why, and that the reason is the model's structure, not
its prices.

The argument
------------
run.py requires, in every hour j,

    sum_i X[i,j] * (1 - lambda_i)  ==  demand[j]        (no slack)

with each X[i,j] in [LB_i, UB_i], and every lignite (thermal-solid)
plant held at one output level all day.  Split delivered supply into
lignite and the rest:

    L          the lignite total, delivered: the same in every hour, and
               anywhere in [L_min, L_max] (the plants' bounds, after
               losses)
    F_j        hydro + gas + solar delivered in hour j: anywhere in
               [F_min, F_max], independently in each hour (no ramp
               limits, and run.py does not apply the hourly solar
               profile, so solar's bounds are the same in every hour)

A schedule exists iff some single L puts demand[j] - L inside
[F_min, F_max] for every hour, ie iff the interval

    [ max(max demand - F_max, L_min),  min(min demand - F_min, L_max) ]

is non-empty.  When the lignite bounds do not bind (they do not, for
any day here) that is simply

    daily range = max demand - min demand
                <=  F_max - F_min = flexible range

The flat lignite block has to sit low enough not to oversupply the
night-time trough and high enough that everything else, flat out, can
lift supply to the evening peak.  The summer days have a deeper trough
relative to their peak than the flexible plants can span, so no level
does both.

Only bounds, losses and demand enter.  Prices, CO2 and the
environmental and sustainability costs set *where* the optimum is, never
*whether* there is one, so the verdict is the same under the 2026 data
(data/2026/), and for every weighting and every algorithm in
run_exps.sh.  We check that rather than assume it: every verdict is
also confirmed by asking the LP itself.

Outputs:
  ../derived/demand_feasibility.csv           every day x data set x
        solar size: daily range, flexible range, the lignite interval,
        the shortfall, and the analytic and LP verdicts
  ../paper/tables/demand_feasibility.tex      tabular: for the PAPER_DAYS,
        the daily range and the shortfall at each solar size (baseline
        data)
  ../paper/figures/demand_feasibility.pdf     the PAPER_DAYS' demand
        against the band that flat lignite plus the flexible plants can
        follow
  ../paper/tables/demand_feasibility_captions.tex   float wrappers and
        captions for the two (captions.tex belongs to sensitivity.py)

Usage:
    python feasibility.py          # a few seconds
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from duals import (SOLAR_SIZES, TEXT_PRIMARY, TEXT_SECONDARY,
                   FIGDIR, TABDIR, DERIVED)

DEMAND_FILE = "../data/demand.csv"
DATA_SETS = ("base", "update_2026")   # keys of run.SCENARIOS
LIGNITE = "thermal-solid"

# the days the paper shows, side by side: the baseline winter day and one
# summer day. the CSV covers every day in demand.csv; the other two
# (09-Jan-26, 14-Jul-21) behave like the winter and summer day shown.
PAPER_DAYS = ("19-Jan-21", "03-Aug-26")

# the figure: baseline data, no new solar (the table covers the rest).
# the summer day is infeasible at every solar size.
FIG_DATA, FIG_SOLAR = "base", 0

# colours: series 1 of the reference palette for demand (the one series
# per panel), series 2 to pick out the hours no flat lignite level can
# reach, and a neutral band, so the band reads as the frame of reference
GRID_GREY = "#e4e3df"
BAND_FILL = "#dcdad3"
DEMAND_BLUE = "#2a78d6"
OUTSIDE_ORANGE = "#eb6834"


def reload_run(solar_size, scenario):
    """Import (or re-import) run.py for a solar size and data set.  run.py
    reads its plant data at import time, driven by sys.argv."""
    sys.argv = ["feasibility.py", "lp", str(solar_size), "0", scenario]
    sys.modules.pop("run", None)
    import run
    return run


def day_label(col):
    return pd.to_datetime(col, format="%d-%b-%y").strftime("%-d %b %Y")


def season(col):
    return ("summer" if pd.to_datetime(col, format="%d-%b-%y").month
            in (6, 7, 8) else "winter")


def flexible_bounds(run):
    """Delivered (after-loss) bounds, kWh per hour, of the lignite total
    and of everything else."""
    pi = run.plant_info
    keep = 1 - run.lambda_i
    lb = 1000 * pi["lower_bound"].values * keep
    ub = 1000 * pi["upper_bound"].values * keep
    coal = (pi["type"] == LIGNITE).values
    return (lb[coal].sum(), ub[coal].sum(),
            lb[~coal].sum(), ub[~coal].sum())


def lp_feasible(run, demand):
    """Ask the LP itself.  lp_solve returns None, rather than falling
    back to another objective, whenever eps is given and the problem is
    not solved to optimality; an infinite bound constrains nothing."""
    saved = run.demand
    run.demand = demand
    try:
        return run.lp_solve(1, 0, 0, 0,
                            eps=(np.inf, None, None, None)) is not None
    finally:
        run.demand = saved


def analyse():
    demand = pd.read_csv(DEMAND_FILE, index_col=0)
    assert len(demand) == 24, demand.shape
    rows = []
    for data in DATA_SETS:
        for solar in SOLAR_SIZES:
            run = reload_run(solar, data)
            L_min, L_max, F_min, F_max = flexible_bounds(run)
            for col in demand.columns:
                d = demand[col].values.astype(float)
                lo = max(d.max() - F_max, L_min)
                hi = min(d.min() - F_min, L_max)
                rows.append({
                    "day": col, "season": season(col), "data": data,
                    "solar_MW": solar,
                    "min_demand_MW": d.min() / 1000,
                    "peak_demand_MW": d.max() / 1000,
                    "daily_range_MW": (d.max() - d.min()) / 1000,
                    "trough_hour": int(d.argmin()),
                    "peak_hour": int(d.argmax()),
                    "flexible_min_MW": F_min / 1000,
                    "flexible_max_MW": F_max / 1000,
                    "flexible_range_MW": (F_max - F_min) / 1000,
                    "lignite_min_MW": L_min / 1000,
                    "lignite_max_MW": L_max / 1000,
                    # the flat lignite levels that work: empty if lo > hi
                    "lignite_level_lo_MW": lo / 1000,
                    "lignite_level_hi_MW": hi / 1000,
                    "shortfall_MW": max(lo - hi, 0.0) / 1000,
                    "feasible": bool(lo <= hi),
                    "lp_feasible": lp_feasible(run, d),
                })
    df = pd.DataFrame(rows)

    # the analysis is exact, so the LP must agree everywhere
    bad = df[df["feasible"] != df["lp_feasible"]]
    if len(bad):
        raise AssertionError("analytic and LP verdicts differ:\n"
                             + bad.to_string())
    # and prices cannot matter: same verdict under either data set
    v = df.pivot_table(index=["day", "solar_MW"], columns="data",
                       values="feasible")
    assert (v["base"] == v["update_2026"]).all(), v
    return df


def write_table(df, filename="demand_feasibility.tex"):
    """Per paper day (baseline data): min and peak demand, the daily
    range, and how far it exceeds the flexible range at each solar
    size."""
    sub = df[df["data"] == "base"]
    days = PAPER_DAYS
    flex = sub.groupby("solar_MW")["flexible_range_MW"].first()

    def fmt(v):
        return f"{v:,.0f}".replace(",", "{,}")

    lines = [
        "% generated by src/feasibility.py -- do not edit",
        "% shortfall: MW by which the daily range exceeds what",
        "% hydro, gas and solar can follow with lignite flat; -- = feasible",
        r"\begin{tabular}{ll" + "r" * (3 + len(SOLAR_SIZES)) + "}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Demand (MW)} & "
        r"\multicolumn{" + str(len(SOLAR_SIZES)) + r"}{c}{Shortfall (MW), "
        r"by new solar (MW)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-" + str(5 + len(SOLAR_SIZES))
        + "}",
        "Day & Season & Min & Peak & Range & "
        + " & ".join(str(s) for s in SOLAR_SIZES) + r" \\",
        r"\midrule",
    ]
    for col in days:
        r = sub[sub["day"] == col].set_index("solar_MW")
        first = r.iloc[0]
        cells = ["--" if r.loc[s, "feasible"] else fmt(r.loc[s,
                 "shortfall_MW"]) for s in SOLAR_SIZES]
        lines.append(" & ".join(
            [day_label(col), first["season"], fmt(first["min_demand_MW"]),
             fmt(first["peak_demand_MW"]), fmt(first["daily_range_MW"])]
            + cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{Flexible range (MW)} & "
        + " & ".join(fmt(flex[s]) for s in SOLAR_SIZES) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path = os.path.join(TABDIR, filename)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def plot(df, filename="demand_feasibility.pdf"):
    """One panel per day: demand, and the band that the flexible plants
    can cover on top of a flat lignite block.  The band has the same
    height all day (the flexible range); the question is whether it can
    be placed so that demand fits inside.  We place it at the middle of
    the interval of workable lignite levels; for an infeasible day that
    interval is empty, and its middle is the level that misses by least
    (by half the shortfall, at both the trough and the peak)."""
    demand = pd.read_csv(DEMAND_FILE, index_col=0)
    sub = df[(df["data"] == FIG_DATA) & (df["solar_MW"] == FIG_SOLAR)] \
        .set_index("day")
    hours = np.arange(24)

    fig, axes = plt.subplots(1, len(PAPER_DAYS), figsize=(9, 3.4),
                             sharex=True, sharey=True, squeeze=False)
    for ax, col in zip(axes.flat, PAPER_DAYS):
        r = sub.loc[col]
        d = demand[col].values / 1000
        L = (r["lignite_level_lo_MW"] + r["lignite_level_hi_MW"]) / 2
        band_lo = L + r["flexible_min_MW"]
        band_hi = L + r["flexible_max_MW"]

        ax.axhspan(band_lo, band_hi, color=BAND_FILL, lw=0, zorder=1)
        ax.plot(hours, d, color=DEMAND_BLUE, lw=2, zorder=3,
                solid_capstyle="round")
        out = (d < band_lo - 1e-6) | (d > band_hi + 1e-6)
        if out.any():
            ax.scatter(hours[out], d[out], s=40, color=OUTSIDE_ORANGE,
                       edgecolor="white", linewidth=1.5, zorder=4)

        verdict = ("feasible" if r["feasible"] else
                   f"infeasible: short by {r['shortfall_MW']:,.0f} MW")
        ax.set_title(f"{day_label(col)} ({r['season']})", loc="left",
                     fontsize=10, color=TEXT_PRIMARY, pad=18)
        ax.text(0, 1.02,
                f"daily range {r['daily_range_MW']:,.0f} MW; {verdict}",
                transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY,
                va="bottom")

        ax.grid(axis="y", color=GRID_GREY, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID_GREY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)

    for ax in axes[-1]:
        ax.set_xlabel("Hour", fontsize=9, color=TEXT_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("MW", fontsize=9, color=TEXT_SECONDARY)

    handles = [
        Line2D([], [], color=DEMAND_BLUE, lw=2, label="Demand"),
        Patch(color=BAND_FILL, label="Reachable"),
        Line2D([], [], ls="", marker="o", markersize=7,
               markerfacecolor=OUTSIDE_ORANGE, markeredgecolor="white",
               label="Hour outside the reachable band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(FIGDIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def write_captions(df, filename="demand_feasibility_captions.tex"):
    base = df[(df["data"] == "base") & (df["solar_MW"] == FIG_SOLAR)]
    flex0 = base["flexible_range_MW"].iloc[0]
    tex = rf"""% generated by src/feasibility.py -- do not edit
\begin{{figure}}
  \centering
  \includegraphics[width=\linewidth]{{demand_feasibility.pdf}}
  \caption{{Why the model has no schedule for a summer day. Each panel is one day's hourly demand (no new solar, baseline data). Lignite output is held constant over the day, so the remaining plants -- hydro, gas and solar -- must follow the whole of the daily range of demand, and between their minimum and maximum outputs they span at most {flex0:,.0f}~MW after losses. The grey band is what is reachable: what they can cover on top of the best flat lignite level. The winter day fits inside it. The summer day has a deeper night-time trough relative to its evening peak, and no lignite level fits: set low enough for the trough, the band cannot reach the peak, and vice versa. Marked hours fall outside the band at that level.}}
  \label{{fig:demand_feasibility}}
\end{{figure}}

\begin{{table}}
  \centering
  \caption{{Feasibility of a winter and a summer demand day. Range is peak minus minimum hourly demand. Shortfall is how far the daily range exceeds the flexible range -- what hydro, gas and solar can follow with lignite held flat -- and ``--'' means a feasible schedule exists. The flexible range grows with new solar because run.py does not cap solar by an hourly production profile; even so, the summer day does not become feasible. The verdicts depend only on plant output bounds, losses and demand, not on prices or emission factors, and are identical under the 2026 data. Each was confirmed by solving the LP.}}
  \label{{tab:demand_feasibility}}
  \input{{demand_feasibility.tex}}
\end{{table}}
"""
    path = os.path.join(TABDIR, filename)
    with open(path, "w") as fh:
        fh.write(tex)
    print("wrote", path)


def main():
    df = analyse()
    path = os.path.join(DERIVED, "demand_feasibility.csv")
    df.to_csv(path, index=False)
    print("wrote", path)
    show = ["day", "data", "solar_MW", "daily_range_MW", "flexible_range_MW",
            "lignite_level_lo_MW", "lignite_level_hi_MW",
            "shortfall_MW", "feasible"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    write_table(df)
    plot(df)
    write_captions(df)


if __name__ == "__main__":
    main()
