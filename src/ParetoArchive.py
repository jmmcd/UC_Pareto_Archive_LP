import random
import numpy as np
import matplotlib.pyplot as plt

class ParetoArchive:
    def __init__(self, f, popsize, generations, random_ind, custom_init_pop, mutate, crossover):
        self.f = f # must return a tuple of numerical obj vals, to be minimised
        self.popsize = popsize
        self.generations = generations
        self.random_ind = random_ind
        self.custom_init_pop = custom_init_pop # can be None, else should return a list of genomes
        self.mutate = mutate
        self.crossover = crossover
        self.cache = {}

    def add_to_pop(self, pop, costs, x):
        tx = tuple(x)
        if tx not in self.cache:
            self.cache[tx] = self.f(x)
            pop.append(x)
            costs.append(self.cache[tx])
        
    def init_pop(self):
        pop = []
        costs = []
        
        # make part of the population using custom_init_pop() -- this
        # allows some population-level control eg for
        # seeding individuals, or for GP ramped half-and-half
        if self.custom_init_pop:
            inds = self.custom_init_pop()
            for x in inds:
                self.add_to_pop(pop, costs, x)
                    
        # then make the rest using random_ind() -- this is the common case
        # where we just need independently sampled initial population
        while len(pop) < self.popsize:
            x = self.random_ind()
            self.add_to_pop(pop, costs, x)
        costs = np.array(costs)
        pop, costs = self.pareto_operator(pop, costs)
        return pop, costs

    def select(self, parents):
        if len(parents) < 6 and random.random() < 0.5:
            # if we don't have many parents, try a random ind sometimes
            return self.random_ind()
        else:
            # maybe use crowding distance sometimes
            return random.choice(parents)
    
    def make_children(self, parents):
        pop = []
        costs = []
        while len(pop) < self.popsize:

            # get a parent
            x = self.select(parents)

            # if we have crossover, get a second, and do it
            if self.crossover:
                y = self.select(parents)
                x = self.crossover(x, y)

            # mutate, either the single parent, or the offspring of crossover
            x = self.mutate(x)

            # calculate fitness, add to cache and pop
            self.add_to_pop(pop, costs, x)
            
        return pop, costs

    def stats(self, gen, costs):
        # FIXME print hypervolume too?
        print(gen, len(costs),
              " ".join("%.3e" % np.min(costs[:, i])
                       for i in range(costs.shape[1])))
        
    def search(self):
        pop, costs = self.init_pop()
        fronts = []
        fronts.append(costs)
        print("Generation PF_size " + " ".join(f"f{i}" for i in range(costs.shape[1])))
        self.stats(0, costs)
        
        for gen in range(1, self.generations):
            # make children and join to current pop
            newpop, newcosts = self.make_children(pop)
            pop = pop + newpop # list concatenate 
            costs = np.concatenate((costs, newcosts)) # numpy concatenate

            # reduce to Pareto front only
            pop, costs = self.pareto_operator(pop, costs)

            # bookkeeping
            self.stats(gen, costs)
            fronts.append(costs)

        return pop, costs, fronts

    def pareto_operator(self, pop, costs):
        assert len(pop) == len(costs)
        is_efficient = pareto_front(costs)
        pop = [pop[i] for i in range(len(pop)) if is_efficient[i]]
        costs = costs[is_efficient]
        assert len(pop) == len(costs)
        return pop, costs
        
def custom_init_pop(n):
    return [
        [0 for _ in range(n)],
        [1 for _ in range(n)]
        ]

def random_bitstring(n):
    return [random.randrange(2) for _ in range(n)]

def onepoint_crossover(x, y):
    # xover returns one individual
    i = random.randrange(1, len(x) - 1)
    # this is a new object, so ok to be mutated later
    return x[:i] + y[i:]

# uniform crossover makes more sense when the correlations among
# genes is not expected to be stronger among "nearby" genes
def uniform_crossover(x, y):
    # this is a new object, so ok to be mutated later
    c = x.copy()
    for i in range(len(x)):
        if random.randrange(2):
            c[i] = y[i]
    return c

def bitflip_mutate(x):
    i = random.randrange(len(x))
    x[i] = 1 - x[i]
    return x

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def pareto_front(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

# this is an old version which makes a nice plot -- one front per generation, all on same plot
# but the make_plots in the Jupyter NB is more suitable when the final front
# just overlays everything else - it puts each front in a facet
def make_plots2(filename, fronts, xlabel=None, ylabel=None):

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(9, 3.5))
    # plot all fronts 
    for i, costs in enumerate(fronts):
        print("doing front", i)
        print(costs)

        if i == len(fronts) - 1:
            # each front is a 2d array of costs. we sort because
            # drawing a line between them needs them sorted
            # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
            costs = costs[costs[:,0].argsort()]
            ls = ""
        else:
            ls = ""
        
        color = (len(fronts) - i)/len(fronts)
        ax[i].plot(costs[:, 0], costs[:, 1],
                 marker="o",
                 #c=np.ones(3) * color,
                 ls=ls,
                 lw=1,
                 #mfc=np.ones(3) * color,
                 mec=np.zeros(3))

    ax[0].set_xlabel(xlabel if xlabel else "x")
    ax[0].set_ylabel(ylabel if ylabel else "y")
    plt.tight_layout()
    plt.savefig(filename)



def ZDT5(x):
    f1 = 1 + sum(x[:30])
    g = 0
    index = 30
    for i in range(10):
        u = sum(x[index:index+5])
        if u < 5:
            g += 2 + u
        else:
            g += 1
        index += 5
    return f1, g / float(f1)

if __name__ == "__main__":
    pa = ParetoArchive(ZDT5, 20, 20,
                       lambda: random_bitstring(80),
                       lambda: custom_init_pop(80),
                       bitflip_mutate, uniform_crossover)
    pop, costs, fronts = pa.search()
    make_plots("ParetoArchive_test.pdf", fronts, "f", "g")
