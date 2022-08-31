import numpy as np

from tqdm import tqdm
from typing import Any, Dict, List, Union, Optional

# Credit: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

def selection(pop: List, scores: List, k: Optional[int] = 3) -> List:
    """Tournament selection procedure that randomly compares k-individuals from
    the population for their scores and then returns the winner/parent

    Parameters
    ----------
    pop: List
    scores: List
    k: int, optional

    Returns
    -------
    parent: List
        The parent from the population that wins the tournament selection
    """
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g., perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(p1: List, p2: List, r_cross: float) -> [List, List]:
    """Takes two parents and the hyperparameter crossover rate to randomly
    determine whether a crossover is performed or not.
    The crossover rate usually lies close to 1.0 and is given in percentage

    Parameters
    ----------
    p1, p2: List
        The two parent lists
    r_cross: float
        The crossover rate

    Returns
    -------
    c1, c2: List
        Two children lists
    """
    # Children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # Check for recombination
    if np.random.rand() < r_cross:
        # Select crossover point that is not on the end of string
        pt = np.random.randint(1, len(p1)-2)
        # Perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return c1, c2

def mutation(bitstring: List, r_mut: float) -> None:
    """This performs the mutation (e.g., randomly flipping bits in the
    bitstring to simulate natural evolution with its random mutations)

    Parameters
    ----------
    bitstring: List
        The bitstring
    r_mut: float
        The mutation rate
    """
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            # flip by bit
            bitstring[i] = 1 - bitstring[i]

def decode(bounds: List, n_bits: int, bitstring: List) -> List:
    """This function decodes the real values from the bitstring, for which it
    uses the bounds of the variables and the numbers of the bits.

    Parameters
    ----------
    bounds: List
        The bounds of the input variables
    n_bits: int
        The number of bits
    bitstring: List
        The bitstring

    Returns
    --------
    decode: List
        The decoded real values
    """
    decoded = list()
    largest = 2**n_bits
    for i, o in enumerate(bounds):
        # Extract the substring
        start, end = i*n_bits, (i*n_bits)+n_bits
        substring = bitstring[start:end]
        # Convert the bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # Convert string to integer
        integer = int(chars, 2)
        # Scale integers to desired range
        value = o[0] + (integer/largest) * (o[1] - o[0])
        # Store
        decoded.append(value)
    return decoded

def genetic_algorithm(objective, bounds: List, n_bits: int,
                      n_iter: int, n_pop: int, r_cross: float,
                      r_mut: float, k: Optional[int] = 3) -> [List, float]:
    """The genetic algorithm is a stochastic global search optimisation
    algorithm inspired by the biological theory of evolution by means of
    natural selection.
    Specifically, the new synthesis that combines an understanding of genetics
    with the theory

    The algorithm uses a genetic representation (bitstrings), fitness
    (functions evaluations), genetic recombination (crossover of bitstrings),
    and mutation (flipping bits)

    First the population of bistrings (candidate solutions) are evaluated for
    fitness (minimisation or maximisation), which is then taken as the fitness
    solution.

    Then, parents are selected based on their fitness by drawing k candidates
    from the population randomly and selecting the member from the group with
    the best fitness (tournament selection).

    Parents are taken in pairs and create two children by using a crossover
    operator, by splittig the both parents (in inverse for the second child,
    called 'onepoint crossover').

    Mutation involves flipping bits in created children candidate solutions.
    Typically the mutation rate is set to 1/L, where L is the length of the
    bitstring.

    ...

    Parameters
    ----------
    objective: function
        The function that is to be minimised
    bounds: List
        The bounds of the function's parameters
    n_bits: int
        The number of bits
    n_iter: int
        The number of iterations to be looped through
    n_pop: int
        The number of the population
    r_cross: float
        The crossover rate
    r_mut: float
        The mutation rate

    Returns
    -------
    best: List
        The 'fittest' bitstring
    score: float
        The fitness score of the bitstring
    """
    # Generate first population randomly
    pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]

    # Keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))

    # Enumerate generations
    print("Running genetic algorithm!")
    for gen in tqdm(range(n_iter)):
        # Decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]

        # Enumerate all candidates in the population
        scores = [objective(c) for c in decoded]

        # Check for the new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]

        # Select parents
        selected = [selection(pop, scores, k) for _ in range(n_pop)]

        # Create the next generation
        children = list()

        for i in range(0, n_pop, 2):
            # Get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # Crossover and mutations
            for c in crossover(p1, p2, r_cross):
                # Mutation
                mutation(c, r_mut)
                # Store for the next generation
                children.append(c)

        # replace population
        pop = children

    return best, best_eval


if __name__ == "__main__":
    ...
