# UC_Pareto_Archive_LP

Solving multi-objective Unit Commitment or Economic Dispatch problems with objective weighting, LP, and Pareto Archive search among weightings, and epsilon-constraint grid search


This code is used in a working paper on viability of decentralised solar energy with a case study in Serbia, James McDermott, Ivan Stevovic, Raziyeh Farmani, Svetlana Stevovic.

## Repo Layout

```
data/           inputs: demand, plant info, solar production
src/            run.py (solver), duals.py, sensitivity.py, ParetoArchive.py
scripts/        run_exps.sh, analysis notebook
runs/           raw output of run.py — NOT in git
paper/figures/  figures for the manuscript (\includegraphics)
paper/tables/   LaTeX table fragments (\input)
derived/        small summary CSVs
```

The split is by what consumes a file, not by which script wrote it.
`runs/` is bulk output — ~19k small `.dat`/`.npy` files, ~180MB — and is
git-ignored, since it is regenerable from `src/` and `data/`. Everything
under `paper/` and `derived/` is small, tracked, and paper-facing.

Some files in `derived/` are also intermediates: `sensitivity.py` writes
`schedule_*.csv` there and the notebook reads them back to make figures,
so the notebook can be run without re-solving.

Reproduce in this order: `scripts/run_exps.sh` (slow, populates `runs/`),
then `src/duals.py` and `src/sensitivity.py`, then the notebook.
