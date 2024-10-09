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
        "plant_info_filename": plant_info_filename
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
             sus_wt
             ):
    """The problem is an LP problem, if we take a single obj.
    Losses are linear. Costs are linear.

    For an approach using many dummy variables - easier to
    understand but maybe slower - see unused.py

    For IP (enforcing rounded decision variables), see unused.py

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

    # effective supply = demand
    for i in range(nplants):
        for j in range(nhours):
            solver.Add(sum(X[i][j] * (1 - lambda_i[i])
                           for i in range(nplants))
                       == demand[j],
                       name=f"effective supply == demand {j}")

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

    # weights
    wts = np.array((technology_cost_wt, emissions_wt,
                    env_wt, sus_wt), dtype=float)
    # a grid search or random search could give all weights = zero,
    # so guard for that:
    if wts.sum() < 10e-7: wts = np.array((1, 1, 1, 1.0)) 
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



def grid_search_lp_wts():
    # consider all combinations of the weights [0, 1, 10, 100, 1000,
    # 10000]
    wt_vals = [0] + [10**i for i in range(5)]
    costs = []
    xs = []
    for (technology_cost_wt, emissions_wt, env_wt, sus_wt) in itertools.product(wt_vals, wt_vals, wt_vals, wt_vals):
        x, c = lp_solve(technology_cost_wt, emissions_wt,
                        env_wt, sus_wt)
        costs.append(c)
        xs.append(x)
    costs = np.array(costs)
    xs = np.array(xs)
    
    return xs[pareto_front(costs)]


def grid_search_lp_wts2():
    # take wt in 10,000 steps in [0, 1] and use that as prod wt
    # and 1-wt as emissions wt (ignore others as correlated
    # with emissions).

    costs = []
    xs = []
    for wt in np.linspace(0, 1, 10001):
        print(wt)
        x, c = lp_solve(wt, 1-wt, 0, 0)
        if c not in costs:
            costs.append(c)
            xs.append(x)
    costs = np.array(costs)
    xs = np.array(xs)
    
    return xs[pareto_front(costs)]



def pareto_archive_lp_wts(popsize, gens):
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
        x, c = lp_solve(*x)
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
        x, c = lp_solve(*ind)
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
        
    elif algo == "grid_search":
        xs = grid_search_lp_wts()
    elif algo == "grid_search2":
        xs = grid_search_lp_wts2()
    elif algo == "pareto_archive":
        xs = pareto_archive_lp_wts(1000, 10)
    elif algo == "random_search":
        xs = pareto_archive_lp_wts(10000, 1)
    else:
        raise ValueError
    
    for i, x in enumerate(xs):
        analyse_save_ind(x,
                         (run_id + "_%d" % i),
                         basedir)
    
