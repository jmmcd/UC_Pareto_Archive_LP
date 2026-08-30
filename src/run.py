import sys
import numpy as np
import pandas as pd
import itertools
import random

# Google OR-Tools. Use pip install ortools
from ortools.linear_solver import pywraplp 

# in current directory
from ParetoArchive import ParetoArchive, uniform_crossover

#################################################################
#
# Read in and set up numerical data
#
#################################################################

# Excel cell references refer to
# "Project revised costs and losses.xlsx"
# which I used to check my implementations were correct.

solar_size = int(sys.argv[2])
plant_info_filename = "../data/plant_info_%d.csv" % solar_size

plant_info = pd.read_csv(plant_info_filename) # Sheet1!A1:I51
nplants = len(plant_info["name"])
nhours = 24

# lower and upper bounds. notice our LB/UB are in kWh, not MWh
LB = 1000 * plant_info["lower_bound"].values
LB = np.broadcast_to(LB, (nplants, nhours)).astype(float).T
UB = 1000 * plant_info["upper_bound"].values
UB = np.broadcast_to(UB, (nplants, nhours)).astype(float).T

# useful in calculating relative values
UB_per_plant = 1000 * plant_info["upper_bound"].values 

# solar data
solar_data = pd.read_csv("../data/solar_production_and_CO2_per_hour.csv")
solar_production_per_hour = solar_data["production"].values
solar_CO2_per_hour = solar_data["CO2"].values

for i in range(len(UB)):
    if plant_info["type"][i] == "solar":
        # solar plants' production is scaled by a factor in [0, 1]
        # according to hour.  we apply this here to scale the max
        # production for any solar plants.  But according to email
        # from Svetlana, we should just ignore this.
        
        # UB[i] = UB[i] * solar_production_per_hour
        pass

# CO2 data
CO2_per_plant = plant_info["CO2"].values.reshape((-1, 1))
CO2_per_plant_per_hour = np.tile(CO2_per_plant, (1, nhours))
for i in range(len(CO2_per_plant)):
    if plant_info["type"][i] == "solar":
        CO2_per_plant_per_hour[i] = solar_CO2_per_hour

# demand data

# Problem!C38:Z38
demand = pd.read_csv("../data/demand.csv",
                     names=["demand"])["demand"].values 

# environmental and sustainability costs
env_cost_per_plant = plant_info["environment_cost"].values.reshape((-1, 1))
sus_cost_per_plant = plant_info["sustainability_cost"].values.reshape((-1, 1))

# losses
k = 0.001408141 # loss factor Sheet1!C53

# loss factor times distance: Sheet1!D28:D51
lambda_i = k * plant_info["distance"].values 

# production costs
production_cost_per_plant = plant_info["production_cost"].values.reshape((-1, 1))





#################################################################
#
# we define f(X) to validate a solution X and calculate
# the objectives, returning a dict of info
#
#################################################################



def f(X):
    """X is a solution as an array of shape (24x24,)
    representing supply per plant per hour"""
    X = X.reshape((nplants, nhours))

    supply = X

    # impose the constraint that all thermal-solid produciton is
    # constant. This is not needed in LP, because whatever constraints
    # we want are imposed by constraints, not here in f.
    
    # for i in range(len(X)):
    #     if plant_info["type"][i] == "thermal-solid":
    #         # thermal-solid: no variation allowed, just take initial value as the all-day value
    #         supply[i] = supply[i, 0]

    # rounding. this doesn't exceed capacity after rounding, because
    # the capacity is set as an upper bound already.  for now we do no
    # rounding at all! result matches Problem!C42:Z65
    
    # rounded_supply = np.around(supply, decimals=-3) 
    rounded_supply = supply.copy() 
    # how far are our decision variables from the rounded values?
    dist_from_rounded = np.sum(np.abs(supply - rounded_supply))

    # supply relative to each plant's max. this is the best output
    # for visualising a schedule.
    with np.errstate(all='raise'):
        try:
            relative_supply = rounded_supply / (UB_per_plant.reshape((-1, 1)))
        except FloatingPointError:
            # occurs when we have new solar = 0MW UB. we can just hack it
            # as the relative value comes out at 0 anyway
            tmp_UB_per_plant = UB_per_plant.copy()
            tmp_UB_per_plant[-1] = 1
            relative_supply = rounded_supply / (tmp_UB_per_plant.reshape((-1, 1)))

    # does not match as Excel multiplies rounded production by
    # lossesby total_cost_coef, Problem!DX4:EU27
    losses = rounded_supply * lambda_i.reshape((-1, 1))
    
    # effective supply is not calculated in Excel
    effective_supply = rounded_supply - losses 
    effective_supply_per_hour = np.sum(effective_supply, axis=0)

    # result matches Problem!AB4:AY27
    technology_cost = rounded_supply * production_cost_per_plant 
    technology_cost = np.sum(technology_cost)

    env_cost = np.sum(rounded_supply * env_cost_per_plant)
    sus_cost = np.sum(rounded_supply * sus_cost_per_plant)

    # result matches Problem!CY4:DV27
    emissions = rounded_supply * CO2_per_plant_per_hour 
    emissions = np.sum(emissions)
    
    # max(demand - supply, 0) is 0 if supply meets or exceeds demand,
    # else positive
    supply_shortfall = np.sum(np.maximum(
        demand - effective_supply_per_hour, 0.0))
    # max(supply - demand, 0) is 0 if demand meets or exceeds supply,
    # else positive
    supply_excess = np.sum(np.maximum(
        effective_supply_per_hour - demand, 0.0))
    
    return {
        "supply": supply,
        "rounded_supply": rounded_supply,
        "relative_supply": relative_supply,
        "technology_cost": technology_cost,
        "supply_shortfall": supply_shortfall,
        "supply_excess": supply_excess,
        "emissions": emissions,
        "dist_from_rounded": dist_from_rounded,
        "env_cost": env_cost,
        "sus_cost": sus_cost,
    }

def f_costs(X):
    # run f, and extract our four objectives from the dict
    # and return as a list
    d = f(X)
    k = ["technology_cost", "emissions", "env_cost", "sus_cost"]
    return dict2list(d, k)

def f_single_obj(X, wts):
    # run f and return a single objective value by weighting
    # the objectives according to wts
    s = sum(wts[k] for k in wts)
    d = f(X)
    return sum(wts[k] * d[k] / s for k in wts)

def dict2list(d, k):
    return [d[k] for k in keys]

def analyse_save_ind(xbest, run_id, basedir):
    # analyse a solution xbest and save some information about it.
    xbest = xbest.reshape((nplants, nhours))
    fname = basedir + "xbest_" + run_id + ".csv"
    np.savetxt(fname, xbest, delimiter=",")
    res = f(xbest)
    supply = res.pop("supply")
    fname = basedir + "supply_" + run_id + ".csv"
    np.savetxt(fname, supply)
    rounded_supply = res.pop("rounded_supply")
    fname = basedir + "rounded_supply_" + run_id + ".csv"
    np.savetxt(fname, rounded_supply)
    relative_supply = res.pop("relative_supply")
    fname = basedir + "relative_supply_" + run_id + ".csv"
    np.savetxt(fname, relative_supply)
    fname = basedir + "objvals_" + run_id + ".dat"
    open(fname, "w").write(repr(res) + "\n")
    fname = basedir + "runid_" + run_id + ".dat"
    params = {
        "algo": algo,
        "solar_size": solar_size,
        "seed": seed,
        "plant_info_filename": plant_info_filename,
        # whether the weights were applied to normalised objectives, and
        # if so the scale factors used. objective values in the
        # objvals_ files are in raw units either way.
        "normalise": normalise,
        "scale": (None if scale is None else list(scale)),
    }
    open(fname, "w").write(repr(params) + "\n")



    


###############################################################
#
# Linear programming
#
###############################################################

    


def lp_solve(technology_cost_wt,
             emissions_wt,
             env_wt,
             sus_wt,
             scale=None,
             eps=None,
             return_duals=False
             ):
    """The problem is an LP problem, if we take a single obj.
    Losses are linear. Costs are linear.

    For an approach using many dummy variables - easier to
    understand but maybe slower - see unused.py

    For IP (enforcing rounded decision variables), see unused.py

    scale: if None (the default), the weights multiply the four raw
    objectives, exactly as in the original experiments.  If given, it
    must be a length-4 array of positive scale factors, and each weight
    is divided by its objective's scale before use.  Passing the
    objective ranges (see payoff_table) is what "normalising the
    objectives" means: it makes a weight of 0.5 mean the same thing for
    every objective, regardless of that objective's units.

    Normalisation changes *which* solutions the search finds.  It never
    changes how they are reported: the (PC, Em, EC, SC) returned here
    are always in raw units, so fronts from normalised and unnormalised
    runs are directly comparable.

    eps: if given, a length-4 sequence of upper bounds on the four
    objectives, with None for any objective that is left unconstrained.
    This is the epsilon-constraint method: instead of steering the
    search by moving the weights, we bound an objective directly and let
    the LP find the best value of the others subject to that bound.  The
    bound is normally binding, so sweeping it gives points at a spacing
    we choose, in the objective's own units.

    An epsilon below an objective's ideal value makes the problem
    genuinely infeasible.  In that case we return None rather than
    falling back to a different objective, so that infeasible epsilons
    are visible to the caller instead of producing a misleading point.

    """


    solver = pywraplp.Solver('Serbia',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # X production are the decision variables
    # this also sets the box constraints
    X = [[solver.NumVar(LB[i,j], UB[i,j], name=f'X[{i}, {j}]')
          for j in range(nhours)]
         for i in range(nplants)]

    # set all Thermal to be constant: each equal to first
    for i in range(nplants):
        if plant_info["type"][i] == "thermal-solid":
            for j in range(1, 24):
                solver.Add(X[i][j] == X[i][0],
                           name=f"thermal-solid {i,j} constant")

    # effective supply = demand. we keep references to these
    # constraints so that we can read off their dual values (shadow
    # prices) later -- see duals.py
    demand_constraints = []
    for j in range(nhours):
        c = solver.Add(sum(X[i][j] * (1 - lambda_i[i])
                           for i in range(nplants))
                       == demand[j],
                       name=f"effective supply == demand {j}")
        demand_constraints.append(c)

    # Production cost, Emissions, Environmental Cost, Sustainability
    # Cost
    
    # we create these dummy vars and constrain them to have the
    # values of our four objectives, given decision variables X
    PC = solver.NumVar(0, np.inf, "PC")
    Em = solver.NumVar(0, np.inf, "Em")
    EC = solver.NumVar(0, np.inf, "EC")
    SC = solver.NumVar(0, np.inf, "SC")

    solver.Add(sum(production_cost_per_plant[i,0] * X[i][j]
                   for j in range(nhours)
                   for i in range(nplants)) == PC)
    solver.Add(sum(CO2_per_plant_per_hour[i,j]    * X[i][j]
                   for j in range(nhours)
                   for i in range(nplants)) == Em)
    solver.Add(sum(env_cost_per_plant[i,0]        * X[i][j]
                   for j in range(nhours)
                   for i in range(nplants)) == EC)
    solver.Add(sum(sus_cost_per_plant[i,0]        * X[i][j]
                   for j in range(nhours)
                   for i in range(nplants)) == SC)

    # epsilon constraints: bound an objective directly
    if eps is not None:
        assert len(eps) == 4
        for var, e in zip((PC, Em, EC, SC), eps):
            if e is not None:
                solver.Add(var <= float(e))

    # weights
    wts = np.array((technology_cost_wt, emissions_wt,
                    env_wt, sus_wt), dtype=float)
    # a grid search or random search could give all weights = zero,
    # so guard for that:
    if wts.sum() < 10e-7: wts = np.array((1, 1, 1, 1.0)) 
    if scale is not None:
        # normalise: divide each weight by its objective's scale, so
        # that the weights refer to comparable quantities. we do this
        # before the sum-to-one step, which is unchanged.
        scale = np.asarray(scale, dtype=float)
        assert scale.shape == (4,) and (scale > 0).all()
        wts = wts / scale
    wts /= wts.sum()

    # objective: a weighted sum of the four objectives
    objective = solver.Objective()
    objective.SetCoefficient(PC, wts[0])
    objective.SetCoefficient(Em, wts[1])
    objective.SetCoefficient(EC, wts[2])
    objective.SetCoefficient(SC, wts[3])
    objective.SetOffset(0)
    objective.SetMinimization()

    ### solve
    result = solver.Solve()

    # some possible outcomes
    d = {solver.OPTIMAL: "OPTIMAL",
         solver.INFEASIBLE: "INFEASIBLE",
         # ABNORMAL likely to do with imprecision
         # https://github.com/google/or-tools/issues/1868
         solver.ABNORMAL: "ABNORMAL"
         # there are other results but have never seen them
         }

    if result != solver.OPTIMAL and eps is not None:
        # with epsilon constraints, a non-optimal result means the
        # epsilons are infeasible. say so, rather than silently
        # solving a different problem.
        return None

    if result != solver.OPTIMAL:
        print("system not solved to optimality", result, d[result])
        print(technology_cost_wt, emissions_wt, env_wt, sus_wt)
        objective = solver.Objective()
        # hack: just re-solve with some trivial weights so we can
        # continue
        objective.SetCoefficient(PC, 1.0)
        objective.SetCoefficient(Em, 0)
        objective.SetCoefficient(EC, 0)
        objective.SetCoefficient(SC, 0)
        objective.SetOffset(0)
        objective.SetMinimization()
        result = solver.Solve()

    # print_sensitivity(solver)
        
    # save the solution x and four individual objective values
    x = np.zeros_like(UB, dtype=float)
    for i in range(nplants):
        for j in range(nhours):
            x[i,j] = X[i][j].solution_value()


    PC = PC.solution_value()
    Em = Em.solution_value()
    EC = EC.solution_value()
    SC = SC.solution_value()

    if return_duals:
        # dual value of the hour-j demand constraint = rate of change
        # of the (weighted) objective per unit of extra demand in hour
        # j. one value per hour. note these are in units of the
        # *weighted* objective per kWh, so they are only a money price
        # when wts = (1, 0, 0, 0).
        duals = np.array([c.dual_value() for c in demand_constraints])
        return x, (PC, Em, EC, SC), duals

    return x, (PC, Em, EC, SC)


def print_sensitivity(solver):
    # is this a continuous problem? we can do more sensitivity
    # analysis if so.
    continuous_problem = all(v.Integer() == False for v in solver.variables())

    # the *reduced cost* for a variable is the change in objective
    # coefficient for the variable which would be required to move the
    # location of the optimum
    if continuous_problem:
        for v in solver.variables():
            print(f"{v.name()} = {v.solution_value():.5}; reduced cost {v.reduced_cost():.5}")
        
    # for c, a in zip(solver.constraints(), solver.ComputeConstraintActivities()):
    #     eps = 0.0000001
    #     # a constraint is *binding* if it is actually preventing the
    #     # optimum from improving -- the constraint line goes through
    #     # the optimum. we print a "*" for binding constraints. eg the
    #     # active ingredient constraint is binding.

    #     # the *dual value* aka *shadow price* of a constraint is the
    #     # amount our profit could improve if the RHS of the constraint
    #     # would improve by 1 unit. for non-binding constraints, the
    #     # dual is 0. if we had an extra 1L of active ingredient, we
    #     # would get an extra EUR100 of profit
    #     binding = "* " if abs(a - c.lb()) < eps or abs(a - c.ub()) < eps else "  "

    #     ctxt = " + ".join(f"{c.GetCoefficient(v):.5}*{v.name()}"
    #                       for v in solver.variables())
        
    #     if continuous_problem:
    #         print(f"{binding} {c.name()}: {c.lb():.5} <= {ctxt} = {a:.5} <= {c.ub():.5}; dual {c.DualValue():.5}")
    #     else:
    #         print(f"{binding} {c.name()}: {c.lb():.5} <= {ctxt} = {a:.5} <= {c.ub():.5}")



def payoff_table():
    """Solve each objective on its own, to get the ideal and nadir points.

    Row k of the payoff table is the vector of all four objective values
    at the solution that minimises objective k alone.  The ideal point
    is the diagonal (each objective at its own best); the nadir point is
    estimated as the column-wise worst, which is the standard estimate
    for problems with more than two objectives.

    Returns (ideal, nadir, ranges).  Pass ranges as lp_solve's scale
    argument to normalise the objectives.
    """
    P = []
    for k in range(4):
        wts = [0.0] * 4
        wts[k] = 1.0
        x, costs = lp_solve(*wts)
        P.append(costs)
    P = np.array(P)
    ideal = P.diagonal().copy()
    nadir = P.max(axis=0)
    ranges = nadir - ideal
    # a degenerate objective (constant over the feasible set) would give
    # a zero range and an undefined normalisation; fall back to 1.0
    ranges[ranges <= 0] = 1.0
    return ideal, nadir, ranges


###############################################################
#
# Grid search and metaheuristic search over weights, using Pareto
#
###############################################################

def pareto_front(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    # Fairly fast for many datapoints, less fast for many costs, somewhat readable
    
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient



def grid_search_lp_wts(scale=None):
    # consider all combinations of the weights [0, 1, 10, 100, 1000,
    # 10000]
    wt_vals = [0] + [10**i for i in range(5)]
    costs = []
    xs = []
    for (technology_cost_wt, emissions_wt, env_wt, sus_wt) in itertools.product(wt_vals, wt_vals, wt_vals, wt_vals):
        x, c = lp_solve(technology_cost_wt, emissions_wt,
                        env_wt, sus_wt, scale=scale)
        costs.append(c)
        xs.append(x)
    costs = np.array(costs)
    xs = np.array(xs)
    
    return xs[pareto_front(costs)]


def grid_search_lp_wts2(scale=None):
    # take wt in 10,000 steps in [0, 1] and use that as prod wt
    # and 1-wt as emissions wt (ignore others as correlated
    # with emissions).

    costs = []
    xs = []
    for wt in np.linspace(0, 1, 10001):
        print(wt)
        x, c = lp_solve(wt, 1-wt, 0, 0, scale=scale)
        if c not in costs:
            costs.append(c)
            xs.append(x)
    costs = np.array(costs)
    xs = np.array(xs)
    
    return xs[pareto_front(costs)]



def epsilon_grid_search(nsteps=100, delta=1e-3, endpoint_tol=1e-9):
    """Epsilon-constraint search: sweep a bound instead of a weighting.

    We minimise the other three objectives, normalised and summed,
    subject to an upper bound epsilon on production cost, and sweep
    epsilon over a fine grid from its ideal to its nadir value.

    Summing the other three is reasonable here because emissions,
    environmental cost and sustainability cost are near-collinear (their
    pairwise cosines are all above 0.94), so they do not really trade
    off against each other -- only against production cost.  That keeps
    the sweep one-dimensional; constraining all three separately, as in
    the textbook method, would need a 3D grid of epsilons.

    The epsilon constraint is binding at the optimum, so the points come
    out spaced exactly as we asked for along the production cost axis.
    That is the point of the method: a gap in the resulting front cannot
    be a sampling failure, because we requested a point at every
    epsilon.  A real gap appears instead as a jump in the *other*
    objectives between neighbouring epsilons.

    delta is the augmentation term (as in AUGMECON): a small weight on
    the constrained objective, which rules out weakly-efficient points
    -- solutions where the bound binds but another objective could still
    be improved for free.  Set delta=0 for the plain method.

    endpoint_tol handles the lower endpoint of the sweep.  At epsilon
    equal to the ideal production cost the constraint can be satisfied
    only with equality, so whether the LP is feasible there comes down
    to the solver's feasibility tolerance, and in practice it reports
    infeasible.  Rather than loosening every epsilon, we retry only the
    ones that come back infeasible, with the bound relaxed by this
    relative amount.  At a production cost of order 1e8 a relative
    tolerance of 1e-9 is well under one currency unit, so it cannot
    move the front by anything observable; it just lets the sweep reach
    its own endpoint.
    """
    ideal, nadir, ranges = payoff_table()

    costs = []
    xs = []
    n_retried = 0
    n_infeasible = 0
    for e in np.linspace(ideal[0], nadir[0], nsteps):
        res = lp_solve(delta, 1, 1, 1,
                       scale=ranges,
                       eps=(e, None, None, None))
        if res is None:
            # only the lower endpoint should land here, where the bound
            # is attainable only with equality. relax it by a relative
            # hair and retry; if it is still infeasible, the epsilon
            # really is below the ideal and we skip it.
            res = lp_solve(delta, 1, 1, 1,
                           scale=ranges,
                           eps=(e * (1 + endpoint_tol), None, None, None))
            if res is None:
                n_infeasible += 1
                continue
            n_retried += 1
        x, c = res
        costs.append(c)
        xs.append(x)
    print(f"epsilon sweep: {len(costs)} solved "
          f"({n_retried} needed the endpoint tolerance), "
          f"{n_infeasible} infeasible")

    costs = np.array(costs)
    xs = np.array(xs)
    return xs[pareto_front(costs)]


def pareto_archive_lp_wts(popsize, gens, scale=None):
    # Pareto archive search over weights

    def halfnormal(mu, sigma):
        return mu + np.abs(np.random.normal(0, sigma))
    def init():
        return [halfnormal(0, 10) for _ in range(4)]
    def mutate(x):
        i = random.randrange(len(x))
        if x[i] == 0:
            x[i] += halfnormal(0, 10)
        else:
            x[i] *= halfnormal(0, 10)
        return x
    def wt_fitness(x):
        x, c = lp_solve(*x, scale=scale)
        return c
    def custom_init_pop():
        # want to ensure that each obj is solo-optimised 
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    pa = ParetoArchive(wt_fitness, popsize, gens, init,
                       custom_init_pop, mutate, uniform_crossover)
    pop, costs, fronts = pa.search()
    # save fronts for later plotting, but notice all fronts are not
    # same length so this will be saved as a list of np arrays, so
    # will have to be loaded with np.load("generations.npy",
    # allow_pickle=True)
    np.save("generations.npy", fronts) 

    # get the phenotypes (where pop is genotypes).
    # we re-solve - this is wasteful but it's only at the end of
    # search and is not slow
    xs = []
    for ind in pop:
        x, c = lp_solve(*ind, scale=scale)
        xs.append(x)
    return np.array(xs)



###############################################################
#
# Main
#
###############################################################


    
if __name__ == "__main__":
    # read from argv
    try:
        algo = sys.argv[1]
        solar_size = int(sys.argv[2])
        seed = int(sys.argv[3])
    except:
        print(sys.argv)
        raise ValueError("Usage: python run.py algo solar_size seed")

    # an algo name ending in "_norm" means: normalise the objectives by
    # their ranges before applying the weights. we strip the suffix to
    # get the underlying algorithm, but keep the full name in algo, so
    # that results land in their own directory and can never be
    # confused with the unnormalised runs.
    normalise = algo.endswith("_norm")
    base_algo = algo[:-len("_norm")] if normalise else algo

    run_id = "_".join(map(
        str,
        [
            "algo", algo,
            "solar", solar_size,
            "seed", seed
        ]))

    print(run_id)

    np.random.seed(seed) # seed both numpy and random module
    random.seed(seed)

    basedir = f"../results/{algo}/solar_{solar_size}/"

    if normalise:
        ideal, nadir, scale = payoff_table()
        print("normalising by objective ranges:", scale)
    else:
        scale = None
    
    if algo == "lp":
        # just for a quick test, min prod cost
        for wt in np.linspace(0.13123232323232323, 0.13135353535353536, 100):
            x, costs = lp_solve(wt, 1-wt, 0, 0) 
            # print(x)
            # print(costs)
            fx = f(x)
            print(wt)
            print(fx['technology_cost'], fx['emissions'], fx['env_cost'], fx['sus_cost'])
        sys.exit()
        
    elif base_algo == "grid_search":
        xs = grid_search_lp_wts(scale)
    elif base_algo == "grid_search2":
        xs = grid_search_lp_wts2(scale)
    elif base_algo == "pareto_archive":
        xs = pareto_archive_lp_wts(1000, 10, scale)
    elif base_algo == "random_search":
        xs = pareto_archive_lp_wts(10000, 1, scale)
    elif base_algo == "epsilon_grid_search":
        # this method normalises internally, so it takes no scale
        xs = epsilon_grid_search()
    else:
        raise ValueError
    
    for i, x in enumerate(xs):
        analyse_save_ind(x,
                         (run_id + "_%d" % i),
                         basedir)
    
