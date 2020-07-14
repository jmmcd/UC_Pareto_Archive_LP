#!/bin/bash -e

# this script runs all the LP experiments, using grid search, Pareto
# archive search, and random search over weights.

# we use -e because we are going to create these dirs and write into
# them. we want the user to deliberately delete them before we
# start. if they don't, mkdir will throw an error, and -e will cause
# the script to exit.


mkdir results/pareto_archive
mkdir results/pareto_archive/solar_0
mkdir results/pareto_archive/solar_10
mkdir results/pareto_archive/solar_50
mkdir results/pareto_archive/solar_100
mkdir results/grid_search
mkdir results/grid_search/solar_0
mkdir results/grid_search/solar_10
mkdir results/grid_search/solar_50
mkdir results/grid_search/solar_100
mkdir results/random_search
mkdir results/random_search/solar_0
mkdir results/random_search/solar_10
mkdir results/random_search/solar_50
mkdir results/random_search/solar_100


python run.py pareto_archive 0 0 && mv generations.pdf results/pareto_archive/generations_pareto_archive_0_0.pdf
python run.py pareto_archive 10 0 && mv generations.pdf results/pareto_archive/generations_pareto_archive_10_0.pdf
python run.py pareto_archive 50 0 && mv generations.pdf results/pareto_archive/generations_pareto_archive_50_0.pdf
python run.py pareto_archive 100 0 && mv generations.pdf results/pareto_archive/generations_pareto_archive_100_0.pdf

python run.py random_search 0 0
python run.py random_search 10 0
python run.py random_search 50 0
python run.py random_search 100 0

python run.py grid_search 0 0
python run.py grid_search 10 0
python run.py grid_search 50 0
python run.py grid_search 100 0
