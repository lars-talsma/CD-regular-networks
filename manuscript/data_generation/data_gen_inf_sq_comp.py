"""
Given a set of swap probabilities and network parameters, runs network simulations on multiple processors calling the 
run_sim function to run specific instances of a swap probability and network parameter value. Returns:
    - The sample mean and standard error of the mean of both the virtual neighborhood size and the virtual node degree; 
    - The error required for the steady state algorithm and the maximum absolute difference between sample means for all
      time steps in the steady state window, for all input swap probabilities and network parameters. 
"""
from timeit import default_timer as timer
import numpy as np
import skeleton_inf_sq
start = timer()
test = False
N_sample = 10000

# Network parameters
prob_succ_gen = 1
prob_succ_swap = 1
max_swap_dist = 2

# Investigated (and associated) parameters
time_cutoff, prob_swap = 10, 0.1
number_of_nodes = 2*(max_swap_dist+1)+1
sim_time = 3*time_cutoff
qubits_per_node = 4*time_cutoff  # Represents an infinite memory for an infinite chain
steady_state_window = time_cutoff

# Reset random number seed for reproducibility 
np.random.seed(0)

virtual_node_degree_sample = np.empty((N_sample, number_of_nodes, number_of_nodes, sim_time))
virtual_neighbourhood_size_sample = np.empty((N_sample, number_of_nodes, number_of_nodes, sim_time))

for n in range(N_sample):
    (virtual_link_age_array, virtual_node_degree, virtual_neighbourhood_size, swap_count, link_fail_count) = (
        skeleton_inf_sq.simulation_infinite_square(prob_succ_gen, prob_succ_swap, prob_swap, time_cutoff, 
                                                    qubits_per_node, sim_time, max_swap_dist, number_of_nodes, test))

    # Usually we sample data from node at position [0, 0] (arbitrary choice); now we sample all nodes
    virtual_node_degree_sample[n, :, :, :] = virtual_node_degree[:, :, :]
    virtual_neighbourhood_size_sample[n, :, :, :] = virtual_neighbourhood_size[:, :, :]

virtual_node_degree_mean = np.mean(virtual_node_degree_sample, axis=0)
virtual_node_degree_std_error = np.std(virtual_node_degree_sample, axis=0)/np.sqrt(N_sample)

virtual_neighbourhood_size_mean = np.mean(virtual_neighbourhood_size_sample, axis=0)
virtual_neighbourhood_size_std_error = np.std(virtual_neighbourhood_size_sample, axis=0)/np.sqrt(N_sample)

virtual_node_degree_steady_state = virtual_node_degree_mean[:, :, -1]
virtual_node_degree_steady_state_std_error = virtual_node_degree_std_error[:, :, -1]

virtual_neighbourhood_size_steady_state = virtual_neighbourhood_size_mean[:, :, -1]
virtual_neighbourhood_size_steady_state_std_error = virtual_neighbourhood_size_std_error[:, :, -1]

results = (virtual_neighbourhood_size_steady_state, virtual_neighbourhood_size_steady_state_std_error,
            virtual_node_degree_steady_state, virtual_node_degree_steady_state_std_error)

end = timer()
print(time_cutoff, prob_swap, end - start)

np.save('data/data_inf_sq_comp', results)