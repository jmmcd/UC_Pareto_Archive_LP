"""Alternative scenarios: how do the Pareto fronts move when one input
changes?

Reviewers asked for three one-at-a-time sensitivity studies, each
against the baseline data, not crossed with each other:

  distance    New Solar sits d km from the load, so it loses
              lambda = k * d of its output like every other plant
              (baseline d = 0, fully decentralised): 5, 10, 178.83 km.
  solar_cost  production cost of both solar plants, 6.6 -> 5.7 and
              4.8 EUR c/kWh.
  coal_co2    emissions of every lignite plant, 0.82 -> 1.6 kg/kWh
              (today's poorer coal); gas is unchanged.

The scenarios themselves are defined once, in SCENARIOS in run.py, and
run by scripts/run_exps.sh (RUN_SCENARIOS=true), which reruns every
algorithm the paper reports at every solar size.  Results land in
../runs/scenarios/<scenario>/, mirroring ../runs/.  This script only
reads them back.

For each scenario and solar size we pool every algorithm and seed and
keep the 4D Pareto-efficient points: the "combined front", exactly as the
notebook does for the baseline.

Merit order (see duals.py) explains most of what moves.  Dispatch ranks
plants by effective cost c_i / (1 - lambda_i), so the table of New
Solar's effective cost against the lignite plants', written to
../derived/, is the first thing to look at when reading the figures.

Outputs:
  ../paper/figures/scenario_<study>.pdf             headline: production
        cost vs emissions, baseline and variants overlaid, one panel per
        solar size
  ../paper/figures/scenario_row_<scenario>.pdf      one row: production
        cost against each other objective, by solar size, baseline behind
  ../paper/figures/pareto_front_pairgrid_scenario_<scenario>.pdf
        the full PairGrid, same style as the baseline ones
  ../paper/tables/scenarios_<study>.tex             best value of each
        objective, and its change against the baseline
  ../derived/scenarios_summary.csv                  the same, unrounded,
        plus dispatch at the cheapest point and objective correlations
  ../derived/scenarios_effective_costs.csv          merit-order table

Usage:
    python scenarios.py             # every scenario must have finished
    python scenarios.py --partial   # skip scenarios not yet complete,
                                    # for a look while runs are going;
                                    # writes to $TMPDIR/scenarios_partial/
                                    # rather than ../paper/ and ../derived/
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from duals import (SOLAR_SIZES, TEXT_PRIMARY, TEXT_SECONDARY,
                   FIGDIR, TABDIR, DERIVED)

RUNS = "../runs/"

# objective keys in the objvals_ files, and their display names
OBJ = {"technology_cost": "Production cost",
       "emissions": "Emissions",
       "env_cost": "Environmental cost",
       "sus_cost": "Sustainability cost"}
OBJ_COLS = list(OBJ.values())

# every run directory the paper's combined front is built from; the same
# list as the notebook.  run_exps.sh produces the same set per scenario.
RUN_DIRS = ["grid_search", "grid_search2", "grid_search_norm",
            "epsilon_grid_search",
            "random_search_seed_0", "random_search_seed_1",
            "pareto_archive_seed_0", "pareto_archive_seed_1",
            "pareto_archive_seed_2", "pareto_archive_seed_3",
            "pareto_archive_seed_4"]

# the three studies.  each lists (scenario, legend label) in the order of
# the input value, baseline first; scenario names are keys of
# run.SCENARIOS.
STUDIES = {
    "distance": {
        "title": "New solar distance from load",
        "variants": [("base", "0 km (baseline)"),
                     ("solar_d5", "5 km"),
                     ("solar_d10", "10 km"),
                     ("solar_d178.83", "178.83 km")],
    },
    "solar_cost": {
        "title": "Solar production cost",
        "variants": [("base", "6.6 €c/kWh (baseline)"),
                     ("solar_cost_5.7", "5.7 €c/kWh"),
                     ("solar_cost_4.8", "4.8 €c/kWh")],
    },
    "coal_co2": {
        "title": "Lignite emissions",
        "variants": [("base", "0.82 kg/kWh (baseline)"),
                     ("coal_co2_1.6", "1.6 kg/kWh")],
    },
    "update_2026": {
        "title": "Input data",
        "variants": [("base", "2020 (baseline)"),
                     ("update_2026", "2026 update")],
    },
}

# Colour.  The variants in a study are ordered levels of one input, so
# they get an ordinal one-hue (blue) ramp, light to dark as the input
# moves away from the baseline; the baseline itself is a neutral grey,
# so it reads as the reference rather than as one more level.  The ramps
# are steps of the validated sequential blue, checked with the dataviz
# validator in --ordinal mode on the light surface.  Marker shape repeats
# the level, so the figures survive greyscale print.
BASELINE_COLOUR = "#8a8a86"
ORDINAL_BLUE = {1: ["#256abf"],
                2: ["#5598e7", "#104281"],
                3: ["#6da7ec", "#256abf", "#0d366b"]}
MARKERS = ["o", "s", "^", "D"]

# the existing PairGrids colour by solar size with this palette; keep it
# so the scenario PairGrids can be read against them directly
SOLAR_PALETTE = sns.cubehelix_palette(4, start=.5, rot=-.75, light=0.7,
                                      reverse=True)

# axis scaling (powers of ten), as in the notebook figures
OBJ_EXP = {"Production cost": 8, "Emissions": 7,
           "Environmental cost": 8, "Sustainability cost": 8}
PC_SCALE, EM_SCALE = 10.0 ** OBJ_EXP["Production cost"], \
                     10.0 ** OBJ_EXP["Emissions"]

# plant rows in plant_info / supply files
NEW_SOLAR = "New Solar"


def scenario_root(scenario):
    if scenario == "base":
        return RUNS
    return os.path.join(RUNS, "scenarios", scenario)


def is_complete(scenario):
    root = scenario_root(scenario)
    return all(os.path.isdir(os.path.join(root, rd, f"solar_{s}"))
               and any(f.startswith("objvals_") for f in
                       os.listdir(os.path.join(root, rd, f"solar_{s}")))
               for rd in RUN_DIRS for s in SOLAR_SIZES)


def read_objvals(d):
    rows = []
    for f in os.listdir(d):
        if f.startswith("objvals_"):
            # the files are a dict repr; under numpy 2 the values print
            # as np.float64(...), hence the np in the namespace
            r = eval(open(os.path.join(d, f)).read(), {"np": np})
            # a schedule that does not meet demand is not a solution at
            # all, and one that meets none of it scores 0 everywhere and
            # dominates the whole front. demand is ~1.4e8 kWh/day, so
            # 1 kWh is far above solver noise.
            gap = max(r["supply_shortfall"], r["supply_excess"])
            if gap > 1.0:
                raise ValueError(f"infeasible result ({gap:.3g} kWh "
                                 f"unmatched): {os.path.join(d, f)}")
            rows.append([float(r[k]) for k in OBJ] + [f, d])
    return pd.DataFrame(rows, columns=OBJ_COLS + ["filename", "dirname"])


def pareto_mask(costs):
    """Pareto-efficient rows of an (n, m) array, minimising.  Exact
    duplicates are kept once."""
    eff = np.ones(len(costs), dtype=bool)
    for i, c in enumerate(costs):
        if eff[i]:
            eff[eff] = np.any(costs[eff] < c, axis=1)
            eff[i] = True
    return eff


def combined_front(scenario):
    """Pool every algorithm and seed; keep the Pareto-efficient points,
    per solar size."""
    out = []
    for solar in SOLAR_SIZES:
        frames = []
        for rd in RUN_DIRS:
            algo, _, seed = rd.partition("_seed_")
            df = read_objvals(os.path.join(scenario_root(scenario), rd,
                                           f"solar_{solar}"))
            df["algo"] = algo
            df["seed"] = int(seed or 0)
            frames.append(df)
        pts = pd.concat(frames, ignore_index=True)
        pf = pts[pareto_mask(pts[OBJ_COLS].values)].copy()
        pf["solar"] = solar
        out.append(pf)
    pf = pd.concat(out).sort_values(["solar", "Production cost"])
    pf["scenario"] = scenario
    return pf.reset_index(drop=True)


def supply_of(row):
    """Production (kWh) per plant per hour, for one front point."""
    f = row["filename"].replace("objvals", "rounded_supply") \
                       .replace(".dat", ".csv")
    return np.genfromtxt(os.path.join(row["dirname"], f))


def plant_info_for(scenario, solar):
    """plant_info with the scenario applied, from run.py's own code, so
    there is one definition of what a scenario means."""
    sys.argv = ["scenarios.py", "lp", str(solar), "0", scenario]
    if "run" in sys.modules:
        del sys.modules["run"]
    import run
    pi = run.plant_info.copy()
    pi["lambda"] = run.lambda_i
    pi["effective_cost"] = pi["production_cost"] / (1 - pi["lambda"])
    return pi


###############################################################
# tables
###############################################################

def effective_cost_table(scenarios):
    """New Solar's delivered cost against the lignite plants', per
    scenario.  Independent of solar size except that New Solar has zero
    capacity at 0MW, so we read the 100MW file."""
    rows = []
    for s in scenarios:
        pi = plant_info_for(s, 100)
        ns = pi[pi["name"] == NEW_SOLAR].iloc[0]
        coal = pi[pi["type"] == "thermal-solid"]
        rows.append({
            "scenario": s,
            "new_solar_cost": ns["production_cost"],
            "new_solar_distance": ns["distance"],
            "new_solar_lambda": ns["lambda"],
            "new_solar_effective_cost": ns["effective_cost"],
            "coal_effective_cost_min": coal["effective_cost"].min(),
            "coal_effective_cost_max": coal["effective_cost"].max(),
            "coal_plants_dearer_than_new_solar":
                int((coal["effective_cost"] > ns["effective_cost"]).sum()),
            "coal_CO2": coal["CO2"].iloc[0],
        })
    return pd.DataFrame(rows)


def summarise(fronts, plant_names):
    """One row per (scenario, solar size): the best value of each
    objective on the combined front, what the cheapest schedule
    dispatches, and how collinear the three non-cost objectives still
    are (the epsilon sweep relies on it)."""
    ns = list(plant_names).index(NEW_SOLAR)
    rows = []
    for (s, solar), pf in fronts.groupby(["scenario", "solar"], sort=False):
        r = {"scenario": s, "solar": solar, "n_front": len(pf)}
        for c in OBJ_COLS:
            r[f"min {c}"] = pf[c].min()
            r[f"max {c}"] = pf[c].max()
        cheapest = pf.loc[pf["Production cost"].idxmin()]
        cleanest = pf.loc[pf["Emissions"].idxmin()]
        for tag, row in (("min PC", cheapest), ("min Em", cleanest)):
            x = supply_of(row)
            r[f"New Solar MWh at {tag}"] = x[ns].sum() / 1000
        cc = np.corrcoef(pf[OBJ_COLS[1:]].values.T)
        r["corr Em-Env"], r["corr Em-Sus"], r["corr Env-Sus"] = \
            cc[0, 1], cc[0, 2], cc[1, 2]
        rows.append(r)
    df = pd.DataFrame(rows)
    # change against the baseline at the same solar size
    base = df[df["scenario"] == "base"].set_index("solar")
    for c in OBJ_COLS:
        b = df["solar"].map(base[f"min {c}"])
        df[f"d min {c} (%)"] = 100 * (df[f"min {c}"] / b - 1)
    return df


def fmt_val_pct(v, p):
    if abs(p) < 0.005:
        return f"{v:.3e}"
    return f"{v:.3e} ({p:+.1f}\\%)"


def write_latex(summary, study, filename):
    spec = STUDIES[study]
    labels = dict(spec["variants"])
    sub = summary[summary["scenario"].isin(labels)].copy()
    order = {s: i for i, (s, _) in enumerate(spec["variants"])}
    sub = sub.sort_values(["solar", "scenario"],
                          key=lambda c: c.map(order) if c.name == "scenario"
                          else c)
    body = pd.DataFrame({
        "Solar (MW)": sub["solar"].values,
        "Scenario": [labels[s].replace("€c", r"\euro{}c")
                     .replace(" (baseline)", " (base)")
                     for s in sub["scenario"]],
        **{f"Best {c.split()[0].lower()}":
           [fmt_val_pct(v, p) for v, p in
            zip(sub[f"min {c}"], sub[f"d min {c} (%)"])]
           for c in OBJ_COLS},
        "New solar at min PC (MWh)":
            [f"{v:.0f}" for v in sub["New Solar MWh at min PC"]],
    })
    tex = body.to_latex(index=False, escape=False, column_format="rl" + "r" * 5)
    tex = (f"% generated by src/scenarios.py -- do not edit\n"
           f"% {spec['title']}: best value of each objective on the combined\n"
           f"% front, with its change against the baseline at the same\n"
           f"% solar size\n" + tex)
    with open(filename, "w") as f:
        f.write(tex)


###############################################################
# figures
###############################################################

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.grid(True, color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)


def variant_styles(study):
    variants = STUDIES[study]["variants"]
    ramp = ORDINAL_BLUE[len(variants) - 1]
    styles = {}
    for i, (s, label) in enumerate(variants):
        colour = BASELINE_COLOUR if i == 0 else ramp[i - 1]
        styles[s] = dict(label=label, color=colour, marker=MARKERS[i])
    return styles


def plot_study(fronts, study, filename, indexed_row=False):
    """Headline: production cost vs emissions, one panel per solar size,
    baseline and every variant overlaid.

    The baseline is drawn as markers underneath like every other level,
    and *also* as a thin grey line on top: where a variant's front
    coincides with the baseline (as 5 and 10 km do), the line shows the
    baseline is there rather than missing.

    indexed_row adds a second row with emissions expressed as % above
    that front's own minimum.  When a scenario moves the emissions level
    a long way (lignite CO2 roughly doubles it) the first row is all
    level and no shape; indexing to a common base puts the shapes on one
    scale, without a second y-axis.

    indexed_row="xy" indexes production cost the same way, for a
    scenario that moves both levels (the 2026 update changes every
    price); the indexed row then has its own x-axis.
    """
    styles = variant_styles(study)
    nrows = 2 if indexed_row else 1
    index_x = indexed_row == "xy"
    fig, axes = plt.subplots(nrows, len(SOLAR_SIZES),
                             figsize=(13, 3.4 if nrows == 1 else 6.6),
                             sharex="row" if index_x else True,
                             sharey="row", squeeze=False)

    def pct_above_min(v):
        return 100 * (v / v.min() - 1)

    def ys(pf, row):
        if row == 0:
            return pf["Emissions"] / EM_SCALE
        return pct_above_min(pf["Emissions"])

    def xs(pf, row):
        if row == 1 and index_x:
            return pct_above_min(pf["Production cost"])
        return pf["Production cost"] / PC_SCALE

    for row in range(nrows):
        for ax, solar in zip(axes[row], SOLAR_SIZES):
            for s, st in styles.items():
                pf = fronts[(fronts["scenario"] == s)
                            & (fronts["solar"] == solar)]
                ax.scatter(xs(pf, row), ys(pf, row),
                           s=7, marker=st["marker"], color=st["color"],
                           linewidth=0, label=st["label"], zorder=2)
                if s == "base":
                    base = pf.sort_values("Production cost")
                    ax.plot(xs(base, row),
                            ys(base, row), color=BASELINE_COLOUR, lw=0.8,
                            zorder=3)
            if row == 0:
                ax.set_title(f"{solar}MW new solar", color=TEXT_PRIMARY,
                             fontsize=10)
            if row == 1 and index_x:
                ax.set_xlabel("Production cost above front minimum (%)")
            elif row == nrows - 1 or index_x:
                ax.set_xlabel("Production cost ($\\times 10^8$)")
            style_axes(ax)
    axes[0, 0].set_ylabel("Emissions ($\\times 10^7$)")
    if indexed_row:
        axes[1, 0].set_ylabel("Emissions above front\nminimum (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title=STUDIES[study]["title"],
               loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8, title_fontsize=9,
               markerscale=2.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if nrows == 1 else 0.92))
    fig.savefig(filename)
    plt.close(fig)


def scenario_label(scenario):
    for spec in STUDIES.values():
        for s, label in spec["variants"]:
            if s == scenario:
                return f"{spec['title']}: {label}"
    return scenario


def plot_row(fronts, scenario, filename):
    """One PairGrid row: production cost against each other objective,
    coloured by solar size, with the baseline front in grey behind."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharex=True)
    for ax, c in zip(axes, OBJ_COLS[1:]):
        base = fronts[fronts["scenario"] == "base"]
        ax.scatter(base["Production cost"] / PC_SCALE,
                   base[c] / 10.0 ** OBJ_EXP[c],
                   s=6, lw=0, color="#d6d5cf",
                   label="Baseline (all sizes)", zorder=1)
        for colour, solar in zip(SOLAR_PALETTE, SOLAR_SIZES):
            pf = fronts[(fronts["scenario"] == scenario)
                        & (fronts["solar"] == solar)]
            ax.scatter(pf["Production cost"] / PC_SCALE,
                       pf[c] / 10.0 ** OBJ_EXP[c],
                       s=8, lw=0, color=colour, label=f"{solar}MW", zorder=2)
        ax.set_xlabel("Production cost ($\\times 10^8$)")
        ax.set_ylabel(f"{c} ($\\times 10^{OBJ_EXP[c]}$)")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=scenario_label(scenario),
               loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8, title_fontsize=9,
               markerscale=2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(filename)
    plt.close(fig)


def plot_pairgrid(pf, filename):
    """The full PairGrid, in the same style as the baseline ones in the
    notebook (make_pf_pairgrid), so they can be laid side by side."""
    pf = pf.rename(columns={"solar": "New solar size (MW)"})

    def diagfunc(x, **kws):
        ax = plt.gca()
        if kws["label"] == 0:
            ax.annotate("Best", xy=(0.1, 0.4), xycoords=ax.transAxes)
        ax.axvline(x=x.min(), ymin=0, ymax=0.3, linewidth=4,
                   color=kws["color"])

    g = sns.PairGrid(pf, vars=OBJ_COLS, hue="New solar size (MW)",
                     palette=SOLAR_PALETTE, height=3, aspect=0.85)
    g = g.map_diag(diagfunc)
    g = g.map_offdiag(plt.scatter, alpha=1, lw=0, s=8)
    g = g.add_legend()
    g._legend.set_bbox_to_anchor((0.25, 0.9))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


###############################################################
# main
###############################################################

def main(partial=False):
    wanted = []
    for spec in STUDIES.values():
        for s, _ in spec["variants"]:
            if s not in wanted:
                wanted.append(s)
    missing = [s for s in wanted if not is_complete(s)]
    if missing and not partial:
        raise SystemExit(f"scenarios not complete: {missing} "
                         f"(run scripts/run_exps.sh, or pass --partial)")
    have = [s for s in wanted if s not in missing]
    figdir, tabdir, derived = FIGDIR, TABDIR, DERIVED
    if partial:
        print("SKIPPING incomplete scenarios:", missing)
        figdir = tabdir = derived = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "scenarios_partial")
        os.makedirs(figdir, exist_ok=True)
        print("writing to", figdir)

    eff = effective_cost_table(have)
    eff.to_csv(os.path.join(derived, "scenarios_effective_costs.csv"),
               index=False)
    print(eff.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fronts = pd.concat([combined_front(s) for s in have], ignore_index=True)
    names = pd.read_csv("../data/plant_info_100.csv")["name"].values
    summary = summarise(fronts, names)
    summary.to_csv(os.path.join(derived, "scenarios_summary.csv"),
                   index=False)
    show = ["scenario", "solar", "n_front"] + \
           [f"d min {c} (%)" for c in OBJ_COLS] + \
           ["New Solar MWh at min PC", "New Solar MWh at min Em",
            "corr Em-Env", "corr Em-Sus"]
    print(summary[show].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))

    for study, spec in STUDIES.items():
        if not all(s in have for s, _ in spec["variants"]):
            continue
        plot_study(fronts, study,
                   indexed_row={"coal_co2": "y",
                                "update_2026": "xy"}.get(study, False),
                   filename=
                   os.path.join(figdir, f"scenario_{study}.pdf"))
        write_latex(summary, study,
                    os.path.join(tabdir, f"scenarios_{study}.tex"))

    for s in have:
        if s == "base":
            continue
        plot_row(fronts, s, os.path.join(figdir, f"scenario_row_{s}.pdf"))
        plot_pairgrid(fronts[fronts["scenario"] == s],
                      os.path.join(figdir,
                                   f"pareto_front_pairgrid_scenario_{s}.pdf"))


if __name__ == "__main__":
    main(partial="--partial" in sys.argv[1:])
