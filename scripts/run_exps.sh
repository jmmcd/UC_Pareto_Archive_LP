#!/bin/bash -e

# Runs the LP experiments: grid search, Pareto archive search, and
# random search over the objective weights.
#
# We use -e because the script creates the results directories and
# writes into them. If a directory already exists, mkdir fails and -e
# stops the script, so that an existing set of results is never
# silently overwritten. Delete the directory deliberately before
# re-running an experiment.
#
# Every experiment is off by default. Turn on the ones you want, either
# by editing the flags below or from the command line, e.g.
#
#     RUN_GRID_SEARCH=true ./run_exps.sh
#
# An algo name ending in "_norm" tells run.py to divide each weight by
# that objective's range (from the payoff table) before applying it, so
# that a weight of 0.5 means the same thing for every objective
# regardless of its units. The objective values written out are still in
# raw units, so normalised and unnormalised results are directly
# comparable. They go to their own directories, so they cannot overwrite
# each other.

RUN_PARETO_ARCHIVE=${RUN_PARETO_ARCHIVE:-false}
RUN_GRID_SEARCH=${RUN_GRID_SEARCH:-false}
RUN_GRID_SEARCH2=${RUN_GRID_SEARCH2:-false}
RUN_GRID_SEARCH_NORM=${RUN_GRID_SEARCH_NORM:-false}
RUN_GRID_SEARCH2_NORM=${RUN_GRID_SEARCH2_NORM:-false}
RUN_RANDOM_SEARCH=${RUN_RANDOM_SEARCH:-false}
RUN_EPSILON_GRID_SEARCH=${RUN_EPSILON_GRID_SEARCH:-false}

# run.py lives in src/ and expects to be run from there: it reads
# ../data/ and writes ../runs/. locate ourselves rather than
# depending on the caller's working directory, so that this script can
# be run from anywhere.
cd "$(dirname "$0")/../src"

RUNS=../runs

SOLAR_SIZES="0 10 50 100"
PARETO_ARCHIVE_SEEDS="0 1 2 3 4"
RANDOM_SEARCH_SEEDS="0 1"

# the grid searches are deterministic, so one seed is enough. it is
# still passed to run.py, which records it in the runid_ files.
GRID_SEED=0


# Run a deterministic algorithm once per solar size. Results go to
# $RUNS/<algo>/solar_<size>/.
run_deterministic () {
    local algo=$1
    echo "=== $algo ==="
    mkdir "$RUNS/$algo"
    for solar in $SOLAR_SIZES; do
        mkdir "$RUNS/$algo/solar_$solar"
        python run.py "$algo" "$solar" "$GRID_SEED"
    done
}

# Run a stochastic algorithm once per (seed, solar size). Each seed gets
# its own run directory, $RUNS/<algo>_seed_<seed>/, so that runs
# can be compared across seeds.
run_seeded () {
    local algo=$1
    shift
    local seeds="$*"
    echo "=== $algo ==="
    for seed in $seeds; do
        mkdir "$RUNS/$algo"
        for solar in $SOLAR_SIZES; do
            mkdir "$RUNS/$algo/solar_$solar"
            python run.py "$algo" "$solar" "$seed"
            # the Pareto archive saves the front at each generation to
            # generations.npy in the current directory; keep it, named
            # for the run that produced it
            if [ -f generations.npy ]; then
                mv generations.npy \
                   "$RUNS/$algo/generations_${algo}_${solar}_${seed}.npy"
            fi
        done
        mv "$RUNS/$algo" "$RUNS/${algo}_seed_${seed}"
    done
}


if [ "$RUN_GRID_SEARCH" = true ];       then run_deterministic grid_search;        fi
if [ "$RUN_GRID_SEARCH2" = true ];      then run_deterministic grid_search2;       fi
if [ "$RUN_GRID_SEARCH_NORM" = true ];  then run_deterministic grid_search_norm;   fi
if [ "$RUN_GRID_SEARCH2_NORM" = true ]; then run_deterministic grid_search2_norm;  fi

# epsilon-constraint search: sweeps an upper bound on production cost
# rather than a weighting, minimising the other three objectives subject
# to that bound. the bound is binding at the optimum, so the points come
# out evenly spaced along the production cost axis by construction. it
# normalises internally, so it takes no scale argument.
if [ "$RUN_EPSILON_GRID_SEARCH" = true ]; then
    run_deterministic epsilon_grid_search
fi

if [ "$RUN_PARETO_ARCHIVE" = true ]; then
    run_seeded pareto_archive $PARETO_ARCHIVE_SEEDS
fi
if [ "$RUN_RANDOM_SEARCH" = true ]; then
    run_seeded random_search $RANDOM_SEARCH_SEEDS
fi

echo "done"
