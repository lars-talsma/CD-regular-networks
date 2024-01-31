import skeleton_fin_sq
import numpy as np
from itertools import product


def run_simulation(params):
    """
    Given a simulation time, a steady-state window and the required network parameters, this function runs N_sample
    realizations of network simulation for a specific network parameter and swap probability and returns:
    - The sample mean and standard error of the mean of both the virtual neighborhood size and the virtual node degree;
    - The error required for the steady state algorithm and the maximum absolute difference between sample means for all
      time steps in the steady state window.
    """
    prob_succ_gen = 1
    prob_succ_swap = 1
    max_swap_dist = 2
    number_of_nodes = 2*(max_swap_dist+1)+1
    test = False
        
    time_cutoff = 11
    time = 3*time_cutoff
    qubits_per_node = 4*time_cutoff  # Represents an infinite memory
    steady_state_window = time_cutoff

    realization, prob_swap = params 

    # Set random number seed for reproducibility 
    np.random.seed(int(realization))

    virtual_link_age_array, virtual_node_degree, virtual_neighbourhood_size, swap_count, link_fail_count = skeleton_fin_sq.simulation_finite_square(prob_succ_gen, prob_succ_swap, prob_swap, time_cutoff, qubits_per_node, time, max_swap_dist, number_of_nodes, test)

    return virtual_neighbourhood_size, virtual_node_degree