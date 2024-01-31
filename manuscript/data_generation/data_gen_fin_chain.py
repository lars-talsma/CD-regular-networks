"""
Given a set of swap probabilities and network parameters, runs network simulations on multiple processors calling the 
run_sim function to run specific instances of a swap probability and network parameter value. Returns:
    - The sample mean and standard error of the mean of both the virtual neighborhood size and the virtual node degree; 
    - The error required for the steady state algorithm and the maximum absolute difference between sample means for all
      time steps in the steady state window, for all input swap probabilities and network parameters. 
"""
import multiprocessing
from itertools import product
from timeit import default_timer as timer
import numpy as np
import data_gen_fin_chain_run_sim

prob_swap_sweep_size = 100+1
prob_swap_range = np.arange(0, prob_swap_sweep_size)/(prob_swap_sweep_size-1)

N_sample = 10000
N_sample_range = np.arange(N_sample)

max_swap_dist = 2
number_of_nodes = 2*(max_swap_dist+1)+1

time_cutoff = 11
time = 6*time_cutoff # Note that we run a longer simulation (usually 3*time_cutoff) for convergence reasons
steady_state_window = time_cutoff

start = timer()

if __name__ == '__main__':
    # Take the cartesian product as the parameters because pool.map() takes only one iterable
    params = product(N_sample_range, prob_swap_range)
    pool = multiprocessing.Pool(78)
    results = pool.map(data_gen_fin_chain_run_sim.run_simulation, params, chunksize=1)

results = np.reshape(results, (N_sample, prob_swap_sweep_size, 2, number_of_nodes, time))
virtual_neighbourhood_size = results[:, :, 0, :, :]
virtual_node_degree = results[:, :, 1, :, :]

virtual_neighbourhood_size_mean = np.empty((prob_swap_sweep_size, number_of_nodes, time))
virtual_neighbourhood_size_std_error = np.empty((prob_swap_sweep_size, number_of_nodes, time))
virtual_node_degree_mean = np.empty((prob_swap_sweep_size, number_of_nodes, time))
virtual_node_degree_std_error = np.empty((prob_swap_sweep_size, number_of_nodes, time))
    
for q in range(prob_swap_sweep_size):
    for n in range(number_of_nodes):
        for t in range(time):
            virtual_neighbourhood_size_mean[q, n, t] = np.mean(virtual_neighbourhood_size[:, q, n, t])
            virtual_neighbourhood_size_std_error[q, n, t] = np.std(virtual_neighbourhood_size[:, q, n, t])/np.sqrt(N_sample)
            virtual_node_degree_mean[q, n, t] = np.mean(virtual_node_degree[:, q, n, t])
            virtual_node_degree_std_error[q, n, t] = np.std(virtual_node_degree[:, q, n, t])/np.sqrt(N_sample)

virtual_node_degree_diff = np.empty((prob_swap_sweep_size, number_of_nodes, steady_state_window-1))
virtual_neighbourhood_size_diff = np.empty((prob_swap_sweep_size, number_of_nodes, steady_state_window-1))

# The error is defined as (b-a)/sqrt(N_sample), where a (b) is the minimum (maximum) value of the performance metric
virtual_node_degree_error = np.empty((prob_swap_sweep_size, number_of_nodes))
virtual_node_degree_error[:] = (2*time_cutoff-0)/np.sqrt(N_sample)
virtual_neighbourhood_size_error = np.empty((prob_swap_sweep_size, number_of_nodes))
virtual_neighbourhood_size_error[:] = (2*np.minimum(time_cutoff, (max_swap_dist+1)*(max_swap_dist+2)/2)-0)/np.sqrt(N_sample)

for q in range(prob_swap_sweep_size):
    for n in range(number_of_nodes):
        for k in range(steady_state_window-1):
            virtual_node_degree_diff[q, n, k] = np.abs(virtual_node_degree_mean[q, n, time-steady_state_window]-
                                                        virtual_node_degree_mean[q, n, time-steady_state_window+k+1])
            virtual_neighbourhood_size_diff[q, n, k] = np.abs(virtual_neighbourhood_size_mean[q, n, time-steady_state_window]-
                                                                virtual_neighbourhood_size_mean[q, n, time-steady_state_window+k+1])


virtual_node_degree_steady_state = virtual_node_degree_mean[:, :, -1]
virtual_node_degree_steady_state_std_error = virtual_node_degree_std_error[:, :, -1]

virtual_neighbourhood_size_steady_state = virtual_neighbourhood_size_mean[:, :, -1]
virtual_neighbourhood_size_steady_state_std_error = virtual_neighbourhood_size_std_error[:, :, -1]

data_fin_chain = (virtual_neighbourhood_size_steady_state, virtual_neighbourhood_size_steady_state_std_error, 
        virtual_neighbourhood_size_error, np.amax(virtual_neighbourhood_size_diff, axis=2), virtual_node_degree_steady_state,
        virtual_node_degree_steady_state_std_error, virtual_node_degree_error, np.amax(virtual_node_degree_diff, axis=2))

end = timer()
print(end - start) # Time in seconds

np.save('data/data_fin_chain', data_fin_chain)
