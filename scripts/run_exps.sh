#!/bin/bash -e

# this script runs all the LP experiments, using grid search, Pareto
# archive search, and random search over weights.

# we use -e because we are going to create these dirs and write into
# them. we want the user to deliberately delete them before we
# start. if they don't, mkdir will throw an error, and -e will cause
# the script to exit.

mkdir ../results/pareto_archive
mkdir ../results/pareto_archive/solar_0
mkdir ../results/pareto_archive/solar_10
mkdir ../results/pareto_archive/solar_50
mkdir ../results/pareto_archive/solar_100

python run.py pareto_archive 0 2 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_0_0.npy
python run.py pareto_archive 10 2 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_10_0.npy
python run.py pareto_archive 50 2 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_50_0.npy
python run.py pareto_archive 100 2 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_100_0.npy

mv ../results/pareto_archive ../results/pareto_archive_seed_2


mkdir ../results/pareto_archive
mkdir ../results/pareto_archive/solar_0
mkdir ../results/pareto_archive/solar_10
mkdir ../results/pareto_archive/solar_50
mkdir ../results/pareto_archive/solar_100

python run.py pareto_archive 0 3 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_0_0.npy
python run.py pareto_archive 10 3 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_10_0.npy
python run.py pareto_archive 50 3 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_50_0.npy
python run.py pareto_archive 100 3 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_100_0.npy

mv ../results/pareto_archive ../results/pareto_archive_seed_3


mkdir ../results/pareto_archive
mkdir ../results/pareto_archive/solar_0
mkdir ../results/pareto_archive/solar_10
mkdir ../results/pareto_archive/solar_50
mkdir ../results/pareto_archive/solar_100

python run.py pareto_archive 0 4 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_0_0.npy
python run.py pareto_archive 10 4 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_10_0.npy
python run.py pareto_archive 50 4 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_50_0.npy
python run.py pareto_archive 100 4 && mv generations.npy ../results/pareto_archive/generations_pareto_archive_100_0.npy

mv ../results/pareto_archive ../results/pareto_archive_seed_4



# mkdir ../results/grid_search2
# mkdir ../results/grid_search2/solar_0
# mkdir ../results/grid_search2/solar_10
# mkdir ../results/grid_search2/solar_50
# mkdir ../results/grid_search2/solar_100

# python run.py grid_search2 0 0
# python run.py grid_search2 10 0
# python run.py grid_search2 50 0
# python run.py grid_search2 100 0



# mkdir ../results/random_search
# mkdir ../results/random_search/solar_0
# mkdir ../results/random_search/solar_10
# mkdir ../results/random_search/solar_50
# mkdir ../results/random_search/solar_100

# python run.py random_search 0 1
# python run.py random_search 10 1
# python run.py random_search 50 1
# python run.py random_search 100 1


# mkdir ../results/grid_search
# mkdir ../results/grid_search/solar_0
# mkdir ../results/grid_search/solar_10
# mkdir ../results/grid_search/solar_50
# mkdir ../results/grid_search/solar_100

# python run.py grid_search 0 0
# python run.py grid_search 10 0
# python run.py grid_search 50 0
# python run.py grid_search 100 0
