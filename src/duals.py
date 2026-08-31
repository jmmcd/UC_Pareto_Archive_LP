"""Shadow prices (dual values) of the hourly demand constraints.

The LP in run.py constrains effective (delivered) supply to equal demand
in each of the 24 hours:

    sum_i X[i,j] * (1 - lambda_i)  ==  demand[j]        for j = 0..23

The *dual value* of that constraint is the rate of change of the optimal
objective per unit of extra demand in hour j -- i.e. what it costs the
system to deliver one more kWh in hour j.  There are 24 such constraints,
so we get 24 duals per solve: one per hour, NOT one per plant per hour.
(The per-plant-per-hour quantity is the reduced cost of X[i,j], which is
a different thing -- see print_sensitivity() in run.py.)

Units: the duals are in units of the *weighted* objective per kWh.  They
are a genuine money price only when the weights are (1, 0, 0, 0), i.e.
when we are minimising production cost alone.  Everywhere else they are
"blended objective units" and should not be labelled as a price.

Usage:
    python duals.py            # sweeps solar sizes and weights, writes
                               # ../derived/duals.csv and three figures

"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# run.py reads its plant data at import time, driven by sys.argv, so we
# have to set argv before importing it.  We re-import it once per solar
# size (see reload_run below).
SOLAR_SIZES = (0, 10, 50, 100)

# weight sweep: production cost vs emissions.  The other two objectives
# are near-collinear with emissions (see the notebook), so a 2D sweep
# covers the interesting trade-off.  wt=1.0 is the pure-cost end, where
# the duals are a real money price.
WEIGHT_SWEEP = np.round(np.linspace(0.0, 1.0, 21), 3)

# Output roots, shared with sensitivity.py.  The split is by what
# consumes the file, not by which script wrote it: FIGDIR and TABDIR
# hold what the manuscript \includegraphics and \inputs, DERIVED holds
# machine-readable summaries (some of which the notebook reads back).
# The bulk run output under ../runs/ is git-ignored; all three of these
# are tracked.
FIGDIR = "../paper/figures/"
TABDIR = "../paper/tables/"
DERIVED = "../derived/"


def reload_run(solar_size):
    """Import (or re-import) run.py for a given solar size."""
    sys.argv = ["duals.py", "lp", str(solar_size), "0"]
    if "run" in sys.modules:
        del sys.modules["run"]
    import run
    return run


def effective_costs(run, wts):
    """Per-plant, per-hour cost of one kWh *delivered*, under weights wts.

    A plant's raw coefficients c_i say what one kWh *produced* costs.  To
    deliver one kWh from plant i we must produce 1/(1 - lambda_i) kWh,
    because a fraction lambda_i is lost on the way.  Hence the division.

    Returns an array of shape (nplants, nhours).  It varies by hour only
    because solar plants have an hourly CO2 coefficient.
    """
    w = np.asarray(wts, dtype=float)
    w = w / w.sum()
    # the three per-plant constant coefficients
    const = (w[0] * run.production_cost_per_plant[:, 0]
             + w[2] * run.env_cost_per_plant[:, 0]
             + w[3] * run.sus_cost_per_plant[:, 0])
    cost = const.reshape((-1, 1)) + w[1] * run.CO2_per_plant_per_hour
    return cost / (1 - run.lambda_i).reshape((-1, 1))


def identify_marginal(run, duals, wts, tol=1e-6):
    """Which plant sets the price in each hour?

    At an optimum the dual for hour j equals the effective cost of the
    plant that is free to move in that hour -- the one not pinned at a
    bound.  We recover it by matching the dual against the effective
    costs.  Hours where nothing matches are labelled "mixed": these are
    hours where the thermal-solid constant-output constraint (which
    couples all 24 hours) is redistributing cost, so no single plant
    sets the price on its own.
    """
    eff = effective_costs(run, wts)
    names, types = [], []
    for j, d in enumerate(duals):
        err = np.abs(eff[:, j] - d)
        i = int(np.argmin(err))
        # relative tolerance, since the duals span two orders of magnitude
        if err[i] <= tol * max(1.0, abs(d)):
            names.append(run.plant_info["name"][i])
            types.append(run.plant_info["type"][i])
        else:
            names.append("mixed")
            types.append("mixed")
    return names, types


def interior_flat_unit(run, x, wts):
    """The thermal-solid plant that is free to move, and its effective cost.

    thermal-solid plants are held to a constant output across all 24 hours
    (see run.py), so their 24 hourly variables collapse to a single degree
    of freedom: the daily level.  At most one such plant is normally
    strictly between its bounds -- the marginal baseload unit, the only
    flat plant with headroom to move.

    Note it is *not* a swing or regulating unit in the usual sense: it
    cannot follow load, because it is pinned flat by construction.  Its one
    free variable is the level of the whole day, which is why it cannot
    answer a demand change in a single hour on its own (this produces the
    "mixed" hours in identify_marginal()) and why it fixes the *level* of
    the entire price curve rather than any one hour's price: see
    check_identities().

    Returns (name, effective cost) or ("", nan) if none is interior.
    """
    X = np.asarray(x).reshape((run.nplants, run.nhours))
    eff = effective_costs(run, wts)
    for i in range(run.nplants):
        if run.plant_info["type"][i] != "thermal-solid":
            continue
        if run.LB[i, 0] + 1e-6 < X[i, 0] < run.UB[i, 0] - 1e-6:
            return run.plant_info["name"][i], float(eff[i].mean())
    return "", float("nan")


def check_identities(df, tol=1e-9):
    """Verify the two dual identities the write-up relies on.

    (1) Daily mean.  Whenever a flat thermal-solid unit is interior, the
        *mean* of the 24 hourly duals equals that unit's effective cost:

            mean_h dual[h]  ==  eff_flat

        Because the unit's output is one variable shared by all 24 hours,
        its optimality condition prices the whole day at once rather than
        any single hour.  This holds at every weight, and it is what sets
        the overall level of the price curve.

    (2) Mixed hour.  In the one hour with no other free plant, the dual is
        whatever the identity above leaves over:

            dual[h*]  ==  24 * eff_flat  -  sum_{h != h*} dual[h]

        which is (1) rearranged.  This is why a mixed dual is not a
        technology price and need not lie between any two plants' costs.
    """
    n1 = n2 = 0
    for (solar, wt), g in df.groupby(["solar_size", "prod_cost_wt"]):
        eff = g["flat_unit_eff"].iloc[0]
        if not np.isfinite(eff):
            continue
        scale = tol * max(1.0, abs(eff))

        err1 = abs(g["dual"].mean() - eff)
        assert err1 <= scale, (
            f"daily-mean identity failed at solar={solar}, wt={wt}: "
            f"mean dual {g['dual'].mean()} != eff {eff} (err {err1})")
        n1 += 1

        mixed = g[g["marginal_type"] == "mixed"]
        for _, row in mixed.iterrows():
            others = g[g["hour"] != row["hour"]]["dual"].sum()
            err2 = abs(row["dual"] - (len(g) * eff - others))
            assert err2 <= scale, (
                f"mixed-hour identity failed at solar={solar}, wt={wt}, "
                f"hour={row['hour']}: err {err2}")
            n2 += 1
    print(f"identities verified: {n1} daily-mean, {n2} mixed-hour "
          f"(tolerance {tol:g} relative)")


def write_effective_cost_table(filename="effective_costs.csv"):
    """Per-plant reference table: OFC, distance, loss, delivered cost.

    The shadow prices are costs per kWh *delivered*, while the objective
    coefficients are per kWh *produced*.  One kWh delivered from plant i
    needs 1/(1 - lambda_i) produced, so the price we observe is

        c_i / (1 - lambda_i),      lambda_i = k * distance_i

    This table is the lookup that turns an observed dual back into a plant.
    Costs and distances do not vary with the solar scenario (only New
    Solar's capacity does), so one table covers all of them.
    """
    run = reload_run(SOLAR_SIZES[-1])
    pi = run.plant_info
    tab = pd.DataFrame({
        "name": pi["name"],
        "type": pi["type"],
        "production_cost": pi["production_cost"],
        "distance": pi["distance"],
        "lambda": run.lambda_i,
        "effective_cost": pi["production_cost"].values / (1 - run.lambda_i),
    }).sort_values("effective_cost").reset_index(drop=True)
    path = os.path.join(DERIVED, filename)
    tab.to_csv(path, index=False)
    print("wrote", path)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tab


def sweep():
    """Solve across solar sizes and weights, collecting duals."""
    rows = []
    for solar_size in SOLAR_SIZES:
        run = reload_run(solar_size)
        for wt in WEIGHT_SWEEP:
            wts = (wt, 1 - wt, 0.0, 0.0)
            x, costs, duals = run.lp_solve(*wts, return_duals=True)
            names, types = identify_marginal(run, duals, wts)
            flat_name, flat_eff = interior_flat_unit(run, x, wts)
            for j in range(run.nhours):
                rows.append({
                    "solar_size": solar_size,
                    "prod_cost_wt": wt,
                    "hour": j,
                    "demand_kWh": run.demand[j],
                    "dual": duals[j],
                    "marginal_plant": names[j],
                    "marginal_type": types[j],
                    "technology_cost": costs[0],
                    "emissions": costs[1],
                    "flat_unit": flat_name,
                    "flat_unit_eff": flat_eff,
                })
        print("solved solar_size =", solar_size)
    return pd.DataFrame(rows)


###############################################################
#
# Figures
#
###############################################################

# Categorical palette, assigned to plant type (the entity), in fixed
# order.  Validated for all-pairs colour-vision-deficient separation on a
# light surface; "mixed" is a neutral, not a hue.  The aqua sits below
# 3:1 contrast on white, so every cell also carries a letter code and the
# full table is written to CSV.
TYPE_COLOURS = {
    "hydro":         "#2a78d6",
    "thermal-solid": "#eb6834",
    "solar":         "#1baf7a",
    "thermal-gas":   "#4a3aa7",
    "mixed":         "#8a8a86",
}
TYPE_CODES = {
    "hydro": "H", "thermal-solid": "L", "solar": "S",
    "thermal-gas": "G", "mixed": "·",
}
TYPE_LABELS = {
    "hydro": "Hydro", "thermal-solid": "Solid", "solar": "Solar",
    "thermal-gas": "Gas", "mixed": "Mixed (coupled hours)",
}
TYPE_ORDER = ["hydro", "thermal-solid", "solar", "thermal-gas", "mixed"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def plot_marginal_technology(df, wt=1.0, filename="marginal_technology.pdf"):
    """Which technology sets the price, by hour and solar capacity.

    A categorical heatmap: the quantity of interest is identity, not
    magnitude, so it gets a categorical palette and no colour bar.
    """
    sub = df[df["prod_cost_wt"] == wt]
    grid = sub.pivot(index="solar_size", columns="hour",
                     values="marginal_type")
    grid = grid.reindex(SOLAR_SIZES)

    codes = {t: i for i, t in enumerate(TYPE_ORDER)}
    Z = np.vectorize(codes.get)(grid.values)

    cmap = ListedColormap([TYPE_COLOURS[t] for t in TYPE_ORDER])
    norm = BoundaryNorm(np.arange(len(TYPE_ORDER) + 1) - 0.5,
                        len(TYPE_ORDER))

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # 2px surface gap between cells, per the mark spec
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SOLAR_SIZES), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # direct labels: relief for the low-contrast slot, and they make the
    # figure readable in greyscale / print
    for r in range(Z.shape[0]):
        for c in range(Z.shape[1]):
            ax.text(c, r, TYPE_CODES[grid.values[r, c]],
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(SOLAR_SIZES)))
    ax.set_yticklabels(SOLAR_SIZES, fontsize=8)
    ax.set_xlabel("Hour", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("New solar (MW)", fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    for s in ax.spines.values():
        s.set_visible(False)

    present = [t for t in TYPE_ORDER if (grid.values == t).any()]
    ax.legend(handles=[Patch(facecolor=TYPE_COLOURS[t],
                             label=f"{TYPE_LABELS[t]} ({TYPE_CODES[t]})")
                       for t in present],
              loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=len(present), frameon=False, fontsize=8)

    ax.set_title("Technology setting the marginal price",
                 fontsize=10, color=TEXT_PRIMARY, loc="left", pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close()


def plot_price_curves(df, wt=1.0, filename="price_curves.pdf"):
    """Marginal cost of delivered energy by hour, one line per solar size.

    Only valid as a money price at wt = 1.0 (pure production cost).
    """
    assert wt == 1.0, "duals are only a money price at wt = 1.0"
    sub = df[df["prod_cost_wt"] == wt]

    # sequential ramp: the series are ordered magnitudes of one variable
    # (solar capacity), so light -> dark in a single hue, not categorical
    ramp = ["#a8c9ee", "#7aabe4", "#3d8ada", "#1a5aa8"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, solar in enumerate(SOLAR_SIZES):
        s = sub[sub["solar_size"] == solar].sort_values("hour")
        ax.plot(s["hour"], s["dual"], color=ramp[k], linewidth=2,
                marker="o", markersize=4, markeredgecolor="#fcfcfb",
                markeredgewidth=0.5, label=f"{solar} MW", zorder=2 + k)

    ax.set_xlabel("Hour", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("Marginal cost of delivered energy", fontsize=9,
                  color=TEXT_SECONDARY)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", color="#e8e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8d8d4")
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.legend(title="New solar", frameon=False, fontsize=8,
              title_fontsize=8, loc="upper left")
    ax.set_title("Hourly marginal cost, minimising production cost only",
                 fontsize=10, color=TEXT_PRIMARY, loc="left", pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close()


def plot_front_summary(df, filename="front_summary.pdf"):
    """Collapse each 24-vector of duals to two scalars, along the front.

    24 duals x 21 weights x 4 solar sizes is too much to show directly.
    Both summaries here are deliberately *unit-free*, because the duals
    themselves are in units of the weighted objective: comparing their
    magnitudes across different weights is meaningless (the mean dual
    just tracks the weights, and says nothing about the system).  So we
    plot (a) the number of hours in the day priced by solar, and (b) the
    ratio of peak to off-peak dual, which is the shape of the price
    curve in one number.
    """
    g = df.groupby(["solar_size", "prod_cost_wt"])
    summ = g.apply(lambda s: pd.Series({
        "solar_hours": (s["marginal_type"] == "solar").sum(),
        "peak_ratio": s["dual"].max() / s["dual"].min(),
    }), include_groups=False).reset_index()

    ramp = ["#a8c9ee", "#7aabe4", "#3d8ada", "#1a5aa8"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, lab in zip(
            axes, ["solar_hours", "peak_ratio"],
            ["Hours per day priced by solar", "Peak / off-peak dual ratio"]):
        for k, solar in enumerate(SOLAR_SIZES):
            s = summ[summ["solar_size"] == solar].sort_values("prod_cost_wt")
            ax.plot(s["prod_cost_wt"], s[col], color=ramp[k], linewidth=2,
                    label=f"{solar} MW")
        ax.set_xlabel("Weight on production cost", fontsize=9,
                      color=TEXT_SECONDARY)
        ax.set_ylabel(lab, fontsize=9, color=TEXT_SECONDARY)
        ax.grid(axis="y", color="#e8e8e4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d8d8d4")
        ax.spines["bottom"].set_color("#d8d8d4")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axes[0].legend(title="New solar", frameon=False, fontsize=8,
                   title_fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for d in (FIGDIR, TABDIR, DERIVED):
        os.makedirs(d, exist_ok=True)
    df = sweep()
    df.to_csv(os.path.join(DERIVED, "duals.csv"), index=False)
    print("wrote", os.path.join(DERIVED, "duals.csv"), df.shape)

    check_identities(df)
    write_effective_cost_table()

    plot_marginal_technology(df)
    plot_price_curves(df)
    plot_front_summary(df)
    print("wrote figures to", FIGDIR)
