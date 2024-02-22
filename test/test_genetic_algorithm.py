# Reference: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
#  -> `export MKL_THREADING_LAYER=GNU`
import random
from numpy.random import randint
from numpy.random import rand
import subprocess
from ast import literal_eval
import os, argparse

# objective function
def evaluate(chromosome, trainset, valset, drop_cache):
    os.system(drop_cache)
    aug_ratio        = chromosome[0]
    cache_type       = chromosome[1]
    reuse_factor     = chromosome[2]
    aug_block_num    = chromosome[3]

    output = './' + '_'.join([str(c) for c in chromosome])

    s = subprocess.run(['python3', 'test_model_run.py', '-t', trainset, '-v', valset, '-o', output, '-c', cache_type,
                        '-r', str(aug_ratio), '-f', str(reuse_factor), '-n', str(aug_block_num)], capture_output=True)

    obj_f = -10**9
    if s.returncode == 0:
        obj_f = literal_eval(s.stdout.decode())
    else:
        print("got error:", s.stderr)

    # objective function
    return obj_f

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(low=0, high=len(pop), size=k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()

    # select crossover point that is not on the end of the string
    pt = randint(1, len(p1)-2)
    # perform crossover
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]

    return [c1, c2]

# mutation operator
def mutation(chromosome, r_mut):
    for i in range(len(chromosome)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            chromosome[i] = random.choice(elements[i])

# genetic algorithm
def genetic_algorithm(objective, trainset, valset, population, n_iter, n_pop, p_cross, r_mut, drop_cache):
    # initial population of random bitstring
    pop = population
    # keep track of best solution
    best, best_eval = 0, -10**9
    hist = dict()

    # enumerate generations
    for gen in range(n_iter):
        scores = []    # [objective(c, drop_cache) for c in pop]
        # evaluate all candidates in the population except for the previous experiment set present in the 'hist'
        for c in pop:
            if '_'.join(str(c)) in hist:
                scores.append(hist['_'.join(str(c))])
            else:
                scores.append(objective(c, trainset, valset, drop_cache))

        # check for new best solution
        for i in range(n_pop):
            print(">Gen{}, f({}) = {:.3f}".format(gen, pop[i], scores[i]))

            # update hist
            hist['_'.join([str(c) for c in pop[i]])] = scores[i]
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">Gen{}, new best f({}) = {:.3f}".format(gen,  pop[i], scores[i]))

        print(">Gen{}, current best f({}) = {:.3f}".format(gen,  best, best_eval))

        # select parents
        selected = [selection(pop, scores, k=int(n_pop * 0.2)) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)

        # replace population
        pop = children

    return [best, best_eval]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset", "-t", metavar='T', type=str,
                        nargs='?', default='/data/home/jmirrinae/ILSVRC2012/train', help='trainset path')
    parser.add_argument("--valset", "-v", metavar='V', type=str,
                        nargs='?', default=None, help='validationset path')
    parser.add_argument("--passwd", "-p", metavar='P', type=str,
                        nargs='?', default=None, help='password')
    args = parser.parse_args()

    assert isinstance(args.passwd, str), "Please enter user password"

    '''
    aug_ratio        = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    cache_type       = ['random', 'loss_sample']
    reuse_factor     = [1, 2, 3, 4, 5]
    aug_block_num    = [1, 2, 3]
    '''

    elements = [[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
                ['random', 'loss_sample'],
                [1, 2, 3, 4, 5],
                [1, 2, 3]]

    # define the total iterations
    n_iter = 5
    # define the population size
    n_pop = 20
    # crossover point
    p_cross = 3
    # mutation rate
    r_mut = 1.0 / float(len(elements))
    drop_cache = 'echo "' + args.passwd + '" | sudo -S sh -c "echo 3 > /proc/sys/vm/drop_caches"'

    population = []
    for i in range(n_pop):
        population.append([random.choice(e) for e in elements])
    print(population)

    # perform the genetic algorithm search
    best, score = genetic_algorithm(evaluate, args.trainset, args.valset, population, n_iter, n_pop, p_cross, r_mut,
                                    drop_cache)
    print('Done!')
    print('f(%s) = %f' % (best, score))
