"""Cost sensitivity: how far must a technology's cost move before the
optimal dispatch changes?

For each technology we report the *stability interval*: the range of
multipliers alpha on its production cost over which the baseline optimal
schedule stays optimal, and what happens just past each end.

The interval is found by bisection, and bisection is safe here.  Scaling
one technology's cost by alpha moves the LP's cost vector along a line,
and the set of cost vectors for which a fixed schedule is optimal is a
convex cone -- so the set of alphas for which the baseline stays optimal
is a single interval containing 1.  We test membership directly: the
baseline is still optimal at alpha iff its cost, under the perturbed
coefficients, equals the re-solved optimum (to OPT_RTOL).

Note what we do *not* test: whether the solver returns the same schedule.
The cost-optimal schedule here is nearly degenerate -- several schedules
tie or nearly tie -- so a re-solve can return a different schedule of
identical cost.  Comparing schedules mistakes those ties for flips, and
is not monotone in alpha, so it cannot be bisected.

An earlier version of this script assumed that dispatch is a merit
order, so that a flip can only happen where two technologies' effective
costs cross, and probed only at those crossings.  That is not so: the
lignite plants must run flat all day, which couples the 24 hours, and the
LP trades lignite against *mixes* of other plants across hours.  Those
trade-offs flip at multipliers where no two plants' costs cross.  It
found breakevens 3-10x too far out.  The is_crossing column records, for
each breakeven, whether it does coincide with a pairwise crossing.

Usage:
    python sensitivity.py        # sweeps solar sizes at w = (1, 0, 0, 0),
                                 # writes ../paper/ and ../derived/
"""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# duals.py owns the run.py re-import dance, the effective-cost definition
# and the validated per-technology palette; reuse rather than restate.
from duals import (reload_run, effective_costs, SOLAR_SIZES,
                   TYPE_COLOURS, TYPE_LABELS, TYPE_ORDER,
                   TEXT_PRIMARY, TEXT_SECONDARY,
                   FIGDIR, TABDIR, DERIVED)

# Raw run.py output, read back by summarise_epsilon_front below.
RUNS = "../runs/"

# The weights the sensitivity is computed at.  (1, 0, 0, 0) is the
# pure-production-cost end of the front, where a multiplier on a
# technology's cost is a plain money statement and needs no
# interpretation in blended objective units.
WTS = (1.0, 0.0, 0.0, 0.0)

# How far out to look for a flip, as a multiplier on production cost.
# Beyond this we report "no flip" rather than a number nobody would
# believe: a technology that needs to get 10x cheaper is out of the
# running for reasons no sensitivity table is going to settle.
ALPHA_MIN, ALPHA_MAX = 0.1, 10.0

# Bisection: the baseline counts as still optimal while its cost is within
# OPT_RTOL of the re-solved optimum.  Production cost is ~5e8, so 1e-9 is
# under one unit -- well above solver noise (the gap at alpha = 1 is ~1e-16)
# and far below any real flip, whose cost advantage grows linearly from 0
# at the breakeven.  At 100MW, 1e-9 and 1e-11 give breakevens within 0.001
# percentage points of each other; 1e-7 moves them by up to 0.04.  BISECT_STEPS halvings of log(alpha)
# over [ALPHA_MIN, ALPHA_MAX] locate a breakeven to ~1e-13 relative.
OPT_RTOL = 1e-9
BISECT_STEPS = 50

# How far past a breakeven to solve for the flipped schedule: far enough
# that the baseline is unambiguously no longer optimal, near enough that
# no second flip intervenes.
FLIP_STEP = 1e-3

# The solar cost multiplier used for the schedule comparison figure in the
# notebook.  At 100MW solar's *first* flip is at about -9%, but it moves
# ~250 MWh on a 138 GWh day (52 MWh of it onto solar) and is invisible in a
# schedule heatmap; at -50% solar still runs only 12 of 24 hours.  It
# reaches round-the-clock operation between -72% and -73%, so 0.25 is a
# round number safely inside that plateau, and the contrast is legible.
# That the number has to be this large is itself the finding.
SCHEDULE_ALPHA = 0.25

# The lignite-emissions comparison: the min-emissions schedule with lignite
# at its baseline CO2 and at today's value.  The value itself is defined
# once, in run.SCENARIOS.  Minimising production cost would be no use here:
# CO2 is not in that objective, so the schedule could not move.
COAL_SCENARIO = "coal_co2_1.6"

# The scenario the paper table and figure report.  The breakevens differ
# between solar scenarios (see check_invariance and derived/sensitivity.csv),
# though in every one each technology's nearer breakeven is within 15%.
HEADLINE_SOLAR = 100


###############################################################
#
# Perturbing one technology's cost
#
###############################################################


class scaled_production_cost:
    """Context manager: multiply one technology's production cost by alpha.

    run.py reads its data into module globals at import time, and both
    lp_solve() and f() look the coefficients up at call time, so patching
    the global is enough to perturb the model -- and restoring it is
    enough to undo it.  We always measure *consequences* outside the
    context, i.e. against the true coefficients, so that a flipped
    schedule is scored at real costs rather than counterfactual ones.
    """

    def __init__(self, run, tech, alpha):
        self.run, self.alpha = run, alpha
        self.mask = (run.plant_info["type"] == tech).values

    def __enter__(self):
        self.saved = self.run.production_cost_per_plant.copy()
        pc = self.saved.copy()
        pc[self.mask, 0] *= self.alpha
        self.run.production_cost_per_plant = pc
        return self.run

    def __exit__(self, *exc):
        self.run.production_cost_per_plant = self.saved
        return False


def dispatch(run, wts):
    """Optimal schedule under the current coefficients, shape (plants, hours)."""
    x, _costs = run.lp_solve(*wts)
    return np.asarray(x).reshape((run.nplants, run.nhours))


def dispatch_changed(x, x_base, rtol=1e-6):
    """Has the schedule moved?

    Only for checking that a re-solve at unchanged coefficients
    reproduces the baseline.  It is *not* a test for a flip: the optimum
    is nearly degenerate, so a re-solve at perturbed coefficients can
    return a different schedule of identical cost.  Use still_optimal().
    """
    return np.abs(x - x_base).sum() > rtol * np.abs(x_base).sum()


def weighted_objective(costs, wts):
    """The LP's scalar objective, from the four raw objective values."""
    w = np.asarray(wts, dtype=float)
    return float(np.dot(w / w.sum(), costs))


def still_optimal(run, x_base, wts, rtol=OPT_RTOL):
    """Is x_base optimal under the *current* (possibly perturbed)
    coefficients?  Compares its cost against a fresh optimum, so ties --
    other schedules of equal cost -- do not count as a change."""
    _x, costs = run.lp_solve(*wts)
    opt = weighted_objective(costs, wts)
    f = run.f(x_base)
    mine = weighted_objective([f["technology_cost"], f["emissions"],
                               f["env_cost"], f["sus_cost"]], wts)
    return mine <= opt + rtol * abs(opt)


def breakeven(run, tech, wts, x_base, far):
    """First multiplier on tech's production cost, going from 1 towards
    `far`, at which x_base stops being optimal; None if it is still
    optimal at `far`.  See the module docstring for why bisection is
    valid."""
    def ok(a):
        with scaled_production_cost(run, tech, a):
            return still_optimal(run, x_base, wts)
    if ok(far):
        return None
    lo, hi = 1.0, far                  # ok(lo), not ok(hi)
    for _ in range(BISECT_STEPS):
        mid = np.sqrt(lo * hi)         # bisect in log(alpha)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(hi)


###############################################################
#
# Pairwise crossings -- a diagnostic only
#
###############################################################


def crossing_multipliers(run, tech, wts):
    """Multipliers on tech's production cost at which some plant of tech
    and some plant outside it have equal effective cost.

    For plant i in tech and plant k outside it, c_i(alpha) == c_k gives

        alpha = (c_k * (1 - lambda_i) - rest_i) / (w0 * pc_i)

    where rest_i is plant i's weighted cost excluding production cost.  At
    w = (1, 0, 0, 0) this is just the effective-cost ratio c_k / c_i.

    If dispatch were a pure merit order these would be the only places a
    flip could happen.  It is not (see the module docstring); we use them
    only to label each breakeven as at a crossing or not.
    """
    w = np.asarray(wts, dtype=float)
    w = w / w.sum()
    # hourly mean: effective costs vary by hour only through solar's
    # hourly CO2 coefficient, which carries weight 0 at the cost end
    eff = effective_costs(run, wts).mean(axis=1)

    pc = run.production_cost_per_plant[:, 0]
    lam = run.lambda_i
    prod_term = w[0] * pc / (1 - lam)          # the part alpha scales
    rest = eff - prod_term                     # the part it does not

    inside = (run.plant_info["type"] == tech).values
    alphas = []
    for i in np.flatnonzero(inside):
        if prod_term[i] <= 0:
            continue                           # alpha cannot move this plant
        for k in np.flatnonzero(~inside):
            alphas.append((eff[k] - rest[i]) / prod_term[i])
    return np.unique(np.round(alphas, 12))


def is_crossing(alpha, crossings, rtol=1e-6):
    return bool(np.any(np.abs(crossings / alpha - 1) < rtol))


def describe_flip(run, x, x_base, tol=0.5):
    """What moved, in daily MWh by technology, largest movers first.

    Every technology that moves is listed: the headline mover is usually
    not the one whose cost we perturbed, because a plant entering the
    merit order displaces whatever was marginal, and the losses mean the
    two do not net to zero.
    """
    delta = (x - x_base).sum(axis=1) / 1000.0          # kWh -> MWh per day
    by_type = pd.Series(delta).groupby(run.plant_info["type"]).sum()
    by_type = by_type[by_type.abs() > tol]
    if by_type.empty:
        return ""
    order = by_type.abs().sort_values(ascending=False).index
    return ", ".join(f"{TYPE_LABELS.get(t, t)} {by_type[t]:+.0f}"
                     for t in order)


###############################################################
#
# The sweep
#
###############################################################


def analyse(solar_size, wts=WTS):
    """Stability interval for every technology, at one solar capacity."""
    run = reload_run(solar_size)
    x_base = dispatch(run, wts)

    # self-consistency: alpha = 1 must reproduce the baseline exactly.
    # this catches both a broken context manager and any solver
    # nondeterminism that would make dispatch_changed() meaningless.
    with scaled_production_cost(run, list(TYPE_ORDER)[0], 1.0):
        assert not dispatch_changed(dispatch(run, wts), x_base), \
            "alpha = 1 did not reproduce the baseline schedule"

    base = run.f(x_base)
    techs = [t for t in TYPE_ORDER if (run.plant_info["type"] == t).any()]

    rows = []
    for tech in techs:
        crossings = crossing_multipliers(run, tech, wts)
        row = {
            "solar_size": solar_size,
            "type": tech,
            "n_plants": int((run.plant_info["type"] == tech).sum()),
            "production_cost": float(
                run.plant_info["production_cost"][
                    run.plant_info["type"] == tech].iloc[0]),
            "capacity_MW": float(
                run.plant_info["upper_bound"][
                    run.plant_info["type"] == tech].sum()),
            "dispatched_MWh": float(x_base[
                (run.plant_info["type"] == tech).values].sum() / 1000.0),
        }
        for label, far in (("down", ALPHA_MIN), ("up", ALPHA_MAX)):
            alpha = breakeven(run, tech, wts, x_base, far)
            row[f"alpha_{label}"] = alpha
            row[f"pct_{label}"] = None if alpha is None else 100 * (alpha - 1)
            row[f"crossing_{label}"] = (None if alpha is None
                                        else is_crossing(alpha, crossings))
            if alpha is None:
                row[f"effect_{label}"] = ""
                row[f"moved_MWh_{label}"] = None
                row[f"d_emissions_{label}"] = None
                row[f"d_prod_cost_{label}"] = None
            else:
                step = (1 - FLIP_STEP) if far < 1 else (1 + FLIP_STEP)
                with scaled_production_cost(run, tech, alpha * step):
                    x = dispatch(run, wts)
                # score the flipped schedule at the *true* coefficients
                new = run.f(x)
                row[f"effect_{label}"] = describe_flip(run, x, x_base)
                # energy re-dispatched: half the L1 change, since every
                # MWh that leaves one plant arrives at another
                row[f"moved_MWh_{label}"] = np.abs(x - x_base).sum() / 2000
                row[f"d_emissions_{label}"] = 100 * (
                    new["emissions"] / base["emissions"] - 1)
                row[f"d_prod_cost_{label}"] = 100 * (
                    new["technology_cost"] / base["technology_cost"] - 1)
        rows.append(row)
    print("analysed solar_size =", solar_size)
    return pd.DataFrame(rows)


def check_intervals(df, solar_size=None, n_inner=4):
    """Checks on the stability intervals.

    Structural claims, which hold for any LP: a technology that runs at
    nothing cannot gain by becoming dearer, so it has no upward flip.

    And a check on the solver rather than on our reasoning: the interval
    property (see the module docstring) says the baseline is optimal at
    every multiplier between the two breakevens, so we re-test a few
    interior points.  A failure here would mean the solver's optimum is
    not reliable enough for the bisection to mean anything.
    """
    n = 0
    for _, r in df.iterrows():
        if r["dispatched_MWh"] <= 1e-6:
            assert pd.isna(r["alpha_up"]), (
                f"{r['type']} is not dispatched yet has an upward flip "
                f"at {r['alpha_up']}")
            n += 1
        if not pd.isna(r["alpha_down"]):
            assert ALPHA_MIN <= r["alpha_down"] < 1, r["alpha_down"]
            n += 1
        if not pd.isna(r["alpha_up"]):
            assert 1 < r["alpha_up"] <= ALPHA_MAX, r["alpha_up"]
            n += 1

    sizes = df["solar_size"].unique() if solar_size is None else [solar_size]
    for size in sizes:
        run = reload_run(size)
        x_base = dispatch(run, WTS)
        for _, r in df[df["solar_size"] == size].iterrows():
            lo = ALPHA_MIN if pd.isna(r["alpha_down"]) else r["alpha_down"]
            hi = ALPHA_MAX if pd.isna(r["alpha_up"]) else r["alpha_up"]
            # strictly inside, log-spaced, avoiding the ends themselves
            for a in np.exp(np.linspace(np.log(lo), np.log(hi),
                                        n_inner + 2)[1:-1]):
                with scaled_production_cost(run, r["type"], a):
                    assert still_optimal(run, x_base, WTS), (
                        f"{r['type']}: baseline not optimal at x{a:.6g}, "
                        f"inside its own stability interval")
                n += 1
    print(f"interval checks passed: {n}")


def check_invariance(df, rtol=1e-9):
    """The breakevens do not depend on how much new solar is built.

    The candidate crossings are set by costs and distances alone, and the
    solar scenarios differ only in New Solar's capacity, so the candidate
    *set* is identical across scenarios -- that part is structural, and
    asserted.  Which candidate actually flips could in principle differ,
    since capacities decide whether a reordering moves any energy; that it
    does not is a finding about this system, so it is reported rather than
    asserted.
    """
    same = True
    for tech, g in df.groupby("type"):
        for col in ("alpha_down", "alpha_up"):
            v = g[col].dropna().values
            if len(v) and not np.allclose(v, v[0], rtol=rtol):
                same = False
                print(f"  {tech}: {col} varies with solar size: {v}")
            if len(v) not in (0, len(g)):
                same = False
                print(f"  {tech}: {col} present in some scenarios only")
    print("breakevens invariant across solar sizes:", same)
    return same


###############################################################
#
# Output
#
###############################################################


def fmt_pct(v):
    return "--" if v is None or pd.isna(v) else f"{v:+.1f}%"


def fmt_mwh(v):
    return "--" if v is None or pd.isna(v) else f"{v:.0f}"


def fmt_pct2(v):
    """Consequences are small next to the breakevens, so they get a decimal."""
    return "--" if v is None or pd.isna(v) else f"{v:+.2f}%"


def write_table(df, solar_size, filename="stability_intervals"):
    """The paper table: one row per technology, at one solar capacity."""
    sub = df[df["solar_size"] == solar_size]
    tab = pd.DataFrame({
        "Technology": [TYPE_LABELS.get(t, t) for t in sub["type"]],
        "Cost/kWh": sub["production_cost"].values,
        "Dispatched (MWh/day)": sub["dispatched_MWh"].round(0).values,
        "Cheaper by": [fmt_pct(v) for v in sub["pct_down"]],
        "moves (MWh/day)": [fmt_mwh(v) for v in sub["moved_MWh_down"]],
        "then CO2": [fmt_pct2(v) for v in sub["d_emissions_down"]],
        "shift (MWh/day)": sub["effect_down"].values,
        "Dearer by": [fmt_pct(v) for v in sub["pct_up"]],
        "moves (MWh/day) ": [fmt_mwh(v) for v in sub["moved_MWh_up"]],
        "then CO2 ": [fmt_pct2(v) for v in sub["d_emissions_up"]],
        "shift (MWh/day) ": sub["effect_up"].values,
    })

    path = os.path.join(DERIVED, filename + ".csv")
    tab.to_csv(path, index=False)
    print("wrote", path)
    print(f"\nStability intervals at solar = {solar_size} MW, "
          f"weights {WTS}:\n")
    print(tab.to_string(index=False))

    # LaTeX, for dropping straight into the paper
    tex = tab.drop(columns=["shift (MWh/day)", "shift (MWh/day) "]).to_latex(
        index=False, escape=True,
        caption=(f"How far each technology's production cost must move "
                 f"before the cost-optimal schedule changes at all "
                 f"({solar_size} MW new solar, minimising production cost). "
                 f"``Moves'' is the energy re-dispatched once the schedule "
                 f"has flipped, out of about 138 GWh/day; the CO2 columns "
                 f"give the resulting change in emissions, scored at true "
                 f"costs. ``--'' means no change out to a factor of "
                 f"{ALPHA_MAX:g} in that direction."),
        label="tab:stability", float_format="%.1f")
    with open(os.path.join(TABDIR, filename + ".tex"), "w") as fh:
        fh.write(tex)
    print("wrote", os.path.join(TABDIR, filename + ".tex"))
    return tab


def plot_tornado(df, solar_size, filename="tornado.pdf"):
    """Stability interval per technology, as a tornado.

    Each bar spans the multipliers on that technology's production cost
    over which the optimal schedule does not move; the bar is the *safe*
    region, and its ends are the breakevens.  The axis is the multiplier
    on a log scale, because halving and doubling a cost are equally large
    changes and a linear axis would say otherwise.  An open end (a bar
    running into the axis edge with no cap) means no flip within the
    search range in that direction.

    Bars are coloured by technology, matching duals.py, and every end
    carries its value directly -- both because identity should not rest on
    colour alone, and because the green sits under 3:1 against the surface.

    No title or explanatory note is drawn: the figure is for a paper,
    where that text belongs in the caption.  write_captions() emits one.
    """
    sub = df[df["solar_size"] == solar_size].copy()
    # most fragile at the top.  fragility is the distance to the *nearer*
    # of the two breakevens, not the width of the interval: a bar with one
    # open end has no meaningful width, only a near edge.
    near = np.minimum(
        np.abs(np.log(sub["alpha_down"].astype(float).fillna(ALPHA_MIN))),
        np.abs(np.log(sub["alpha_up"].astype(float).fillna(ALPHA_MAX))))
    sub = sub.assign(_near=near.values).sort_values("_near")

    fig, ax = plt.subplots(figsize=(7.5, 0.62 * len(sub) + 1.9))

    # the axis spans the closed ends with a margin on each side (room for
    # the end labels), and at least +/-5%.  open ends run to the edge and
    # say how far we looked.
    down = sub["alpha_down"].dropna().astype(float).values
    up = sub["alpha_up"].dropna().astype(float).values
    lo_lim = min(0.95, np.exp(1.3 * np.log(down.min()))) if len(down) else 0.95
    hi_lim = max(1.05, np.exp(1.45 * np.log(up.max()))) if len(up) else 1.05
    for y, (_, r) in enumerate(sub.iterrows()):
        open_lo, open_hi = pd.isna(r["alpha_down"]), pd.isna(r["alpha_up"])
        lo = lo_lim if open_lo else r["alpha_down"]
        hi = hi_lim if open_hi else r["alpha_up"]
        colour = TYPE_COLOURS.get(r["type"], "#8a8a86")
        ax.barh(y, hi - lo, left=lo, height=0.52, color=colour,
                edgecolor="#fcfcfb", linewidth=2, zorder=3)

        # direct labels at each closed end; an open end is labelled inside
        # the bar instead, so that no number implies a breakeven we did
        # not find
        if open_lo:
            ax.text(lo * 1.004, y, f"no flip to \u00d7{ALPHA_MIN:g}",
                    ha="left", va="center",
                    fontsize=7.5, color="#fcfcfb", zorder=4)
        else:
            ax.text(lo, y, fmt_pct(r["pct_down"]) + "  ", ha="right",
                    va="center", fontsize=8, color=TEXT_SECONDARY, zorder=4)
        if open_hi:
            ax.text(hi / 1.004, y, f"no flip to \u00d7{ALPHA_MAX:g}",
                    ha="right", va="center",
                    fontsize=7.5, color="#fcfcfb", zorder=4)
        else:
            ax.text(hi, y, "  " + fmt_pct(r["pct_up"]), ha="left",
                    va="center", fontsize=8, color=TEXT_SECONDARY, zorder=4)

    ax.axvline(1.0, color="#8a8a86", linewidth=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(lo_lim, hi_lim)
    pcts = [-50, -30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100]
    ticks = [1 + p / 100 for p in pcts if lo_lim < 1 + p / 100 < hi_lim]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("0" if t == 1 else f"{100 * (t - 1):+.0f}%")
                        .replace("-", "\u2212") for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([TYPE_LABELS.get(t, t) for t in sub["type"]],
                       fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Change in production cost (log scale)", fontsize=9,
                  color=TEXT_SECONDARY)
    ax.grid(axis="x", color="#e8e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close()
    print("wrote", os.path.join(FIGDIR, filename))


def plot_consequences(df, filename="flip_consequences.pdf"):
    """How much the flip is worth, across the four solar scenarios.

    The breakevens themselves do not move with solar capacity (see
    check_invariance), so a tornado per scenario would be four identical
    figures.  What does move is the *consequence*: the change in emissions
    once the flip has happened, scored at true coefficients.  Pairing the
    two is the point -- a breakeven that is close but worth nothing is a
    different finding from one that is far away but decisive.

    Two panels, because "if it got cheaper" and "if it got dearer" are
    different questions and sharing an axis would only invite subtraction.
    The panel titles stay -- they identify which panel is which, and the
    figure is unreadable without them -- but the overall title does not;
    see write_captions().
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    panels = [("down", "If it got cheaper"), ("up", "If it got dearer")]
    sizes = sorted(df["solar_size"].unique())

    for ax, (label, title) in zip(axes, panels):
        ends = {}
        for tech in TYPE_ORDER:
            s = df[df["type"] == tech].sort_values("solar_size")
            if s.empty or s[f"d_emissions_{label}"].isna().all():
                continue
            ax.plot(s["solar_size"], s[f"d_emissions_{label}"],
                    color=TYPE_COLOURS[tech], linewidth=2, marker="o",
                    markersize=5, markeredgecolor="#fcfcfb",
                    markeredgewidth=1.2, label=TYPE_LABELS[tech], zorder=3)
            end = (s["solar_size"].iloc[-1], s[f"d_emissions_{label}"].iloc[-1])
            ends.setdefault((end[0], round(end[1], 4)), []).append(tech)
        # direct labels at the right-hand end, so identity is never
        # colour-alone.  several technologies often flip to the *same*
        # neighbouring schedule, so their lines end on one point: one
        # joined label there, rather than overprinted ones.
        for (xe, ye), techs in ends.items():
            ax.annotate(", ".join(TYPE_LABELS[t] for t in techs), (xe, ye),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color=TEXT_SECONDARY, va="center")
        ax.axhline(0, color="#8a8a86", linewidth=1)
        ax.set_title(title, fontsize=9, color=TEXT_PRIMARY, loc="left", pad=6)
        ax.set_xlabel("New solar (MW)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_xticks(sizes)
        ax.set_xlim(-8, max(sizes) * 1.55)
        ax.grid(axis="y", color="#e8e8e4", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#d8d8d4")
        ax.spines["bottom"].set_color("#d8d8d4")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    axes[0].set_ylabel("Change in emissions at the flip (%)", fontsize=9,
                       color=TEXT_SECONDARY)
    # one shared legend, below the panels: inside either one it would sit
    # on top of a series
    seen = {}
    for ax in axes:                       # hydro appears in one panel only
        h, l = ax.get_legend_handles_labels()
        seen.update({lab: hnd for lab, hnd in zip(l, h) if lab not in seen})
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=len(seen), frameon=False,
               fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, filename), bbox_inches="tight")
    plt.close()
    print("wrote", os.path.join(FIGDIR, filename))


def write_schedules(solar_size=HEADLINE_SOLAR, alpha=SCHEDULE_ALPHA):
    """Baseline and cheap-solar schedules, for the notebook to plot.

    The notebook is analysis-only -- it never imports ortools -- so the two
    schedules are solved here and written in the same format run.py uses
    for its own relative_supply files, which is what the notebook's
    existing heatmap cell already reads.

    Both panels come from this one function so that they are guaranteed to
    differ only in the solar cost.  check_baseline_matches_front() confirms
    the baseline is the same optimum as the stored min-production-cost
    point, i.e. the schedule the paper already shows.
    """
    run = reload_run(solar_size)
    pct = round(100 * (alpha - 1))
    out = {}
    for tag, a in (("baseline", 1.0), (f"solar{pct:+d}pct", alpha)):
        with scaled_production_cost(run, "solar", a):
            x = dispatch(run, WTS)
        rel = run.f(x)["relative_supply"]
        path = os.path.join(DERIVED, f"schedule_{tag}_{solar_size}MW.csv")
        np.savetxt(path, rel)          # run.py's format: space-delimited
        out[tag] = (path, x)
        print("wrote", path)

    solar = (run.plant_info["type"] == "solar").values
    for tag, (_path, x) in out.items():
        xs = x[solar]
        print(f"  {tag:16s} solar {xs.sum() / 1000:6.0f} MWh/day, "
              f"{int((xs.sum(axis=0) > 1e-6).sum()):2d} of 24 hours, "
              f"emissions {run.f(x)['emissions']:.4e}")
    return out


def min_emissions_dispatch(run, rtol=1e-9):
    """Lexicographic min-emissions schedule, shape (plants, hours).

    Minimise emissions; then, holding emissions at that optimum, minimise
    production cost.  Minimising emissions alone can leave ties, and the
    solver may return a weakly-efficient one that costs more for no
    emissions gain.  rtol relaxes the bound by a hair, for the same
    reason as epsilon_grid_search's endpoint_tol: a bound that can only
    be met with equality is at the mercy of the feasibility tolerance.
    """
    _x, costs = run.lp_solve(0.0, 1.0, 0.0, 0.0)
    res = run.lp_solve(1.0, 0.0, 0.0, 0.0,
                       eps=(None, costs[1] * (1 + rtol), None, None))
    assert res is not None, "second stage infeasible"
    x, _ = res
    return np.asarray(x).reshape((run.nplants, run.nhours))


def write_coal_schedules(solar_size=HEADLINE_SOLAR, scenario=COAL_SCENARIO):
    """Min-emissions schedules with lignite CO2 at baseline and in the
    given scenario, for the notebook to plot (same format as
    write_schedules), plus a per-technology summary.

    Both schedules are scored under *both* CO2 values, so the table shows
    how much of the change in emissions is the coefficient alone (same
    schedule, dearer coal) and how much is re-dispatch.
    """
    runs = {"base": reload_run(solar_size)}
    runs[scenario] = reload_run(solar_size, scenario)
    co2 = {s: r.plant_info.loc[r.plant_info["type"] == "thermal-solid",
                               "CO2"].iloc[0] for s, r in runs.items()}
    xs = {s: min_emissions_dispatch(r) for s, r in runs.items()}

    rows = []
    for s, x in xs.items():
        r = runs[s]
        tag = f"min_em_coal-{co2[s]:g}"
        path = os.path.join(DERIVED, f"schedule_{tag}_{solar_size}MW.csv")
        np.savetxt(path, r.f(x)["relative_supply"])
        print("wrote", path)
        row = {"schedule": tag, "solved with lignite CO2": co2[s]}
        for t in TYPE_ORDER[:-1]:
            m = (r.plant_info["type"] == t).values
            row[f"{TYPE_LABELS[t]} MWh/day"] = x[m].sum() / 1000
        row["production cost"] = r.f(x)["technology_cost"]
        for s2, r2 in runs.items():
            row[f"emissions at CO2={co2[s2]:g}"] = r2.f(x)["emissions"]
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(DERIVED, f"min_emissions_coal_{solar_size}MW.csv")
    df.to_csv(path, index=False)
    print("wrote", path)
    print(df.T.to_string(float_format=lambda v: f"{v:.4g}"))
    return df


def check_baseline_matches_front(solar_size=HEADLINE_SOLAR, tol=1e-4):
    """The baseline panel is the schedule the paper already shows.

    The min-production-cost end of the stored epsilon sweep is the same LP
    optimum we solve here, so the comparison figure's left panel should
    reproduce the existing min_prod_cost figure rather than quietly being
    a slightly different vertex.  Skips, with a message, if the sweep
    results are not present.
    """
    import ast
    import glob

    pattern = (f"{RUNS}epsilon_grid_search/solar_{solar_size}/"
               f"objvals*.dat")
    files = glob.glob(pattern)
    if not files:
        print("no stored front for solar =", solar_size, "-- check skipped")
        return None

    best, best_file = np.inf, None
    for f in files:
        d = ast.literal_eval(open(f).read().strip())
        if d["technology_cost"] < best:
            best, best_file = d["technology_cost"], f

    stored = np.genfromtxt(best_file.replace("objvals", "relative_supply")
                                    .replace(".dat", ".csv"))
    run = reload_run(solar_size)
    ours = run.f(dispatch(run, WTS))["relative_supply"]
    err = float(np.abs(stored - ours).max())
    assert err <= tol, (f"baseline schedule differs from the stored "
                        f"min-production-cost point by {err:g}")
    print(f"baseline matches stored min-production-cost schedule "
          f"(max |diff| {err:.1e} in relative output)")
    return err


def write_captions(solar_size, invariant, filename="captions.tex"):
    """LaTeX figure blocks for the two figures.

    The figures carry no titles or notes of their own, so everything a
    reader needs to interpret them is here instead.  Emitting the captions
    from the same place that sets ALPHA_MAX and the scenario keeps them
    from drifting out of step with the figures they describe.
    """
    blocks = [(
        "tornado.pdf", "fig:tornado",
        f"Cost change needed to move the optimal schedule "
        f"({solar_size}~MW new solar, minimising production cost). "
        f"Each bar spans the changes in that technology's production "
        f"cost over which the baseline schedule remains optimal; its ends "
        f"are the breakevens, annotated as percentage changes, beyond "
        f"which a different schedule is strictly cheaper. The axis is "
        f"logarithmic, so that a fall and a rise by the same factor are "
        f"equally far from the centre. A bar running to the plot edge, "
        f"marked ``no flip'', means the schedule does not change in that "
        f"direction out to a factor of {ALPHA_MAX:g}. "
        f"Technologies are ordered by their nearer breakeven, most "
        f"fragile first. "
        + ("The breakevens are identical in all four solar scenarios."
           if invariant else
           "The breakevens differ somewhat between solar scenarios; "
           "see the supplementary table.")
    ), (
        "flip_consequences.pdf", "fig:consequences",
        f"What the first flip is worth. For each technology and each "
        f"direction, the change in total emissions once the schedule has "
        f"flipped at the breakeven of Figure~\\ref{{fig:tornado}}, "
        f"evaluated at true cost coefficients, against the amount of new "
        f"solar capacity built. Left: the technology becomes cheaper. "
        f"Right: it becomes dearer. Absent series have no breakeven in "
        f"that direction. Read together with Figure~\\ref{{fig:tornado}}: "
        f"a breakeven that is close by is not necessarily one that matters."
    )]

    tex = "\n".join(
        "\\begin{figure}\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{f}}}\n"
        f"  \\caption{{{cap}}}\n"
        f"  \\label{{{lab}}}\n"
        "\\end{figure}\n"
        for f, lab, cap in blocks)
    path = os.path.join(TABDIR, filename)
    with open(path, "w") as fh:
        fh.write(tex)
    print("wrote", path)


if __name__ == "__main__":
    for d in (FIGDIR, TABDIR, DERIVED):
        os.makedirs(d, exist_ok=True)

    df = pd.concat([analyse(s) for s in SOLAR_SIZES], ignore_index=True)
    df.to_csv(os.path.join(DERIVED, "sensitivity.csv"), index=False)
    print("wrote", os.path.join(DERIVED, "sensitivity.csv"), df.shape)

    check_intervals(df, solar_size=HEADLINE_SOLAR)
    invariant = check_invariance(df)
    write_table(df, solar_size=HEADLINE_SOLAR)
    plot_tornado(df, solar_size=HEADLINE_SOLAR)
    plot_consequences(df)
    write_captions(HEADLINE_SOLAR, invariant)

    check_baseline_matches_front()
    write_schedules()
    write_coal_schedules()
